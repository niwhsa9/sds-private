#pragma once
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
#include <curand.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "sds_core.cuh"
#include "tensor.cuh"

namespace sds
{

// concept Cost = ... TODO(amg)

template <typename T, typename... Dims>
Tensor<T, sizeof...(Dims)> random_tensor(Dims... dims)
{
  Tensor<T, sizeof...(Dims)> tensor(dims...);
  std::mt19937 rng(42);
  std::normal_distribution<T> gauss(0.f, 1.f);
  for (int i = 0; i < tensor.numel(); ++i) tensor.data()[i] = gauss(rng);
  return tensor;
}

template <typename T, int dim>
Tensor<T, dim + 1> random_batch_tensor(
    const TensorView<T, dim>& mean, const TensorView<T, dim>& std_dev,
    int batch_size)
{
  // Tensor<T, dim + 1> tensor(batch_size, mean.shape());
  std::array<int, dim + 1> shape;
  shape[0] = batch_size;
  std::ranges::copy(std::span{mean.shape_}, shape.begin() + 1);
  Tensor<T, dim + 1> tensor(shape);

  std::mt19937 rng(42);
  for (int b = 0; b < batch_size; ++b)
    for (int i = 0; i < mean.numel(); ++i)
    {
      std::normal_distribution<T> gauss(mean.data()[i], std_dev.data()[i]);
      tensor.template slice<0>(b).data()[i] = gauss(rng);
    }
  return tensor;
}

// inplace version of random batch tensor
template <typename T, int dim>
void random_batch_tensor_inplace(
    TensorView<T, dim + 1> tensor, const TensorView<T, dim>& mean,
    const TensorView<T, dim>& std_dev)
{
  std::mt19937 rng(42);
  for (int b = 0; b < tensor.shape(0); ++b)
    for (int i = 0; i < mean.numel(); ++i)
    {
      std::normal_distribution<T> gauss(mean.data()[i], std_dev.data()[i]);
      tensor.template slice<0>(b).data()[i] = gauss(rng);
    }
}

__device__ inline float quadratic_target_cost(
    const TensorView<float, 1>& x_target, const TensorView<float, 1>& Qf,
    const TensorView<float, 1>& Q, const TensorView<float, 1>& Rf,
    const TensorView<float, 2>& x_seq, const TensorView<float, 2>& u_seq,
    float dt = 0.0)
{
  // final quadratic cost to target
  float cost = 0.0f;
  for (int i = 0; i < x_seq.shape(1); ++i)
  {
    float diff = x_seq(x_seq.shape(0) - 1, i) - x_target(i);
    cost += diff * diff * Qf(i);
  }
  // running control cost
  for (int t = 0; t < u_seq.shape(0); ++t)
  {
    for (int i = 0; i < u_seq.shape(1); ++i)
    {
      float u = u_seq(t, i);
      cost += Rf(i) * u * u * dt;
    }
  }
  // running state cost
  for (int t = 0; t < x_seq.shape(0) - 1; ++t)
  {
    for (int i = 0; i < x_seq.shape(1); ++i)
    {
      float diff = x_seq(t, i) - x_target(i);
      cost += diff * diff * Q(i) * dt;
    }
  }
  return cost;
}

//__device__ float quaternion_cost(
//    Eigen::Quaterniond<float, 1> q_target, const TensorView<float, 2> x_seq)
//{
//  //
//}

template <typename T>
void cem_mean_std(
    TensorView<T, 2> mean, TensorView<T, 2> std_dev, const TensorView<T, 3>& u,
    const TensorView<T, 1>& costs, int num_elites)
{
  int batch_size = u.shape(0);
  int horizon = u.shape(1);
  int n_u = u.shape(2);

  std::vector<int> indices(batch_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(), indices.begin() + num_elites, indices.end(),
      [&](int a, int b) { return costs(a) < costs(b); });

  for (int t = 0; t < horizon; ++t)
    for (int j = 0; j < n_u; ++j)
    {
      mean(t, j) = 0.0;
      for (int k = 0; k < num_elites; ++k) mean(t, j) += u(indices[k], t, j);
      mean(t, j) /= float(num_elites);
    }

  for (int t = 0; t < horizon; ++t)
    for (int j = 0; j < n_u; ++j)
    {
      std_dev(t, j) = 0.0;
      for (int k = 0; k < num_elites; ++k)
      {
        float diff = u(indices[k], t, j) - mean(t, j);
        std_dev(t, j) += diff * diff;
      }
      std_dev(t, j) = std::sqrt(std_dev(t, j) / float(num_elites) + 1e-6f);
    }
}

// One thread per rollout — evaluates CostFunc on that rollout's x_seq/u_seq
// slices and writes the scalar result into costs[b].
template <typename Cost>
__global__ void cost_kernel(
    Cost cost, TensorView<float, 3> x_seq,  // [N, T+1, n_x]
    TensorView<float, 3> u_seq,             // [N, T,   n_u]
    TensorView<float, 1> costs)             // [N]
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= x_seq.shape(0)) return;
  // 2-D views for this rollout — fix batch dim (0) at b.
  auto x_b = x_seq.slice<0>(b);  // [T+1, n_x]
  auto u_b = u_seq.slice<0>(b);  // [T,   n_u]
  costs(b) = cost(x_b, u_b);
}

template <
    RolloutProvider<float> P, typename Cost,
    typename PreemptFn = std::nullptr_t>
void cem(
    TensorView<float, 2> u_mean, TensorView<float, 2> u_std_dev,
    TensorView<float, 3> u_samples, TensorView<float, 1> costs, P& plant,
    const TensorView<float, 1>& x0, const Cost& cost, int T, int n_u, float dt,
    int n_samples = 512, int n_elites = 64, int n_iters = 100,
    PreemptFn preempt_callback = nullptr)
{
  for (int iter = 0; iter < n_iters; ++iter)
  {
    random_batch_tensor_inplace(u_samples, u_mean, u_std_dev);
    auto x_samples = plant(x0, u_samples, dt);  // [N, T+1, n_x]

    cost_kernel<<<(n_samples + 255) / 256, 256>>>(
        cost, x_samples, u_samples, costs);
    cudaDeviceSynchronize();

    cem_mean_std(u_mean, u_std_dev, u_samples, costs, n_elites);

    if constexpr (!std::is_same_v<PreemptFn, std::nullptr_t>)
    {
      if (preempt_callback())
      {
        std::cerr << "CEM Preempted at iteration " << iter << std::endl;
        break;
      }
    }
  }
}

template <
    RolloutProvider<float> P, typename Cost,
    typename PreemptFn = std::nullptr_t>
Tensor<float, 2> cem(
    P& plant, const TensorView<float, 1>& x0, const Cost& cost, int T, int n_u,
    float dt, int n_samples = 512, int n_elites = 64, int n_iters = 100,
    float sigma_init = 0.5f,
    std::optional<TensorView<float, 2>> warm_start = std::nullopt,
    PreemptFn preempt_callback = nullptr)
{
  Tensor<float, 2> u_mean(T, n_u);
  u_mean.fill(0.0f);
  if (warm_start.has_value())
  {
    auto warm_start_view = warm_start.value();
    if (warm_start_view.shape(0) != T || warm_start_view.shape(1) != n_u)
      throw std::runtime_error(
          "Warm start shape does not match T and n_u in cem");
    u_mean.deep_copy_from(warm_start_view);
  }

  Tensor<float, 2> u_std_dev(T, n_u);
  u_std_dev.fill(sigma_init);

  Tensor<float, 3> u_samples(n_samples, T, n_u);
  Tensor<float, 1> costs(n_samples);

  cem(u_mean.view(), u_std_dev.view(), u_samples, costs, plant, x0, cost, T,
      n_u, dt, n_samples, n_elites, n_iters, preempt_callback);
  return u_mean;
}

struct CEMData
{
  CEMData(int T, int n_u, int n_samples)
      : u_mean(T, n_u), u_std_dev(T, n_u), u_samples(n_samples, T, n_u),
        costs(n_samples)
  {
  }

  Tensor<float, 2> u_mean, u_std_dev;
  Tensor<float, 3> u_samples;
  Tensor<float, 1> costs;
};

// Linear interpolation on a uniformly spaced physical-time tape. The mean
// extrapolates; standard deviations hold the endpoints, as in interpolate().
inline float interpolate_control_tape(const TensorView<float, 2>& tape,
                                      float knot_dt, float t, int j,
                                      bool hold_endpoints = false)
{
  float p = t / knot_dt;
  if (hold_endpoints)
    p = std::clamp(p, 0.0f, static_cast<float>(tape.shape(0) - 1));
  int k = static_cast<int>(
      std::floor(std::clamp(p, 0.0f, static_cast<float>(tape.shape(0) - 2))));
  float w = p - k;
  return (1.0f - w) * tape(k, j) + w * tape(k + 1, j);
}

// Refit on physical time, excluding elites past their last control sample.
// dts and samples are host-readable (the CEM buffers use managed memory).
inline float cem_variable_time_mean_std(TensorView<float, 2> mean,
                                        TensorView<float, 2> std_dev,
                                        const TensorView<float, 3>& samples,
                                        const TensorView<float, 1>& dts,
                                        const TensorView<float, 1>& costs,
                                        int n_elites, float& dt_mean,
                                        float& dt_std)
{
  std::vector<int> elites(samples.shape(0));
  std::iota(elites.begin(), elites.end(), 0);
  std::partial_sort(elites.begin(), elites.begin() + n_elites, elites.end(),
                    [&](int a, int b) { return costs(a) < costs(b); });
  elites.resize(n_elites);
  float max_dt = 0.0f;
  double dt_sum = 0.0;
  for (int b : elites)
  {
    max_dt = std::max(max_dt, dts(b));
    dt_sum += dts(b);
  }
  dt_mean = static_cast<float>(dt_sum / n_elites);
  double dt_var = 0.0;
  for (int b : elites) dt_var += (dts(b) - dt_mean) * (dts(b) - dt_mean);
  dt_std = std::max(static_cast<float>(std::sqrt(dt_var / n_elites)), 1e-4f);

  // The tape has T controls at 0,...,(T-1)*dt, not T+1 states.
  for (int k = 0; k < mean.shape(0); ++k)
  {
    // Use double for coverage comparisons so the longest elite is included
    // at the final knot despite float rounding.
    double t = static_cast<double>(k) * max_dt;
    for (int j = 0; j < mean.shape(1); ++j)
    {
      int count = 0;
      double m = 0.0, m2 = 0.0;
      for (int b : elites)
      {
        if (t > static_cast<double>(samples.shape(1) - 1) * dts(b)) continue;
        float value = interpolate_control_tape(samples.slice<0>(b), dts(b),
                                               static_cast<float>(t), j);
        // Welford population variance avoids cancellation near a constant
        // tape.
        double delta = value - m;
        m += delta / ++count;
        m2 += delta * (value - m);
      }
      mean(k, j) = static_cast<float>(m);
      std_dev(k, j) =
          std::max(static_cast<float>(std::sqrt(std::max(m2 / count, 0.0))),
                   count < 3 ? 1.0f : 1e-2f);
    }
  }
  return max_dt;
}

template <typename Cost>
__global__ void variable_time_cost_kernel(Cost cost,
                                          TensorView<float, 3> x_seq,
                                          TensorView<float, 3> u_seq,
                                          TensorView<float, 1> dts,
                                          TensorView<float, 1> costs)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < x_seq.shape(0))
    costs(b) = cost(x_seq.slice<0>(b), u_seq.slice<0>(b), dts(b));
}

struct VariableTimeCEMResult
{
  Tensor<float, 2> u_mean;
  float dt;
};

// Cost signature: cost(x_seq, u_seq, candidate_dt). Existing cem() is
// unchanged.
template <VariableTimeRolloutProvider<float> P, typename Cost,
          typename PreemptFn = std::nullptr_t>
VariableTimeCEMResult cem_variable_time(
    P& plant, const TensorView<float, 1>& x0, const Cost& cost, int T,
    int n_u, float dt_mean, float dt_std = 0.01f, int n_samples = 512,
    int n_elites = 64, int n_iters = 100, float sigma_init = 0.5f,
    std::optional<TensorView<float, 2>> warm_start = std::nullopt,
    PreemptFn preempt_callback = nullptr)
{
  if (T < 2 || n_u < 1 || n_samples < 1 || n_elites < 1 ||
      n_elites > n_samples || n_iters < 0 || !std::isfinite(dt_mean) ||
      dt_mean < 1e-4f || !std::isfinite(dt_std) || dt_std < 0.0f ||
      !std::isfinite(sigma_init) || sigma_init < 0.0f)
    throw std::invalid_argument(
        "Invalid variable-time CEM dimensions or distribution");
  Tensor<float, 2> mean(T, n_u), std_dev(T, n_u);
  mean.fill(0.0f);
  std_dev.fill(sigma_init);
  if (warm_start)
  {
    if (warm_start->shape(0) != T || warm_start->shape(1) != n_u)
      throw std::invalid_argument(
          "Variable-time CEM warm start shape mismatch");
    mean.deep_copy_from(*warm_start);
  }
  Tensor<float, 3> samples(n_samples, T, n_u);
  Tensor<float, 1> dts(n_samples), costs(n_samples);
  float knot_dt = dt_mean;
  // Seed once per solve, rather than replaying the same noise each iteration.
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (int iter = 0; iter < n_iters; ++iter)
  {
    for (int b = 0; b < n_samples; ++b)
    {
      dts(b) = std::max(dt_mean + dt_std * normal(rng), 1e-4f);
      for (int k = 0; k < T; ++k)
        for (int j = 0; j < n_u; ++j)
        {
          float t = k * dts(b);
          samples(b, k, j) =
              interpolate_control_tape(mean, knot_dt, t, j) +
              interpolate_control_tape(std_dev, knot_dt, t, j, true) *
                  normal(rng);
        }
    }
    auto x_samples = plant(x0, samples.view(), dts.view());
    variable_time_cost_kernel<<<(n_samples + 255) / 256, 256>>>(
        cost, x_samples.view(), samples.view(), dts.view(), costs.view());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Invalid rollouts cannot silently win elite selection (NaNs break
    // sorting).
    for (int b = 0; b < n_samples; ++b)
      if (!std::isfinite(costs(b)))
        costs(b) = std::numeric_limits<float>::infinity();
    int n_finite = 0;
    for (int b = 0; b < n_samples; ++b) n_finite += std::isfinite(costs(b));
    if (n_finite < n_elites)
      throw std::runtime_error("Too few finite variable-time CEM candidates");
    knot_dt = cem_variable_time_mean_std(
        mean.view(), std_dev.view(), samples.view(), dts.view(), costs.view(),
        n_elites, dt_mean, dt_std);
    if constexpr (!std::is_same_v<PreemptFn, std::nullptr_t>)
      if (preempt_callback()) break;
  }
  Tensor<float, 2> controls(T, n_u);
  for (int k = 0; k < T; ++k)
    for (int j = 0; j < n_u; ++j)
      controls(k, j) =
          interpolate_control_tape(mean, knot_dt, k * dt_mean, j);
  return {std::move(controls), dt_mean};
}

}  // namespace sds
