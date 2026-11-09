#pragma once
#include <cmath>
#include <concepts>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "tensor.cuh"

namespace sds
{

template <typename Sys>
concept DynamicalSystem =
    // std::floating_point<typename Sys::ScalarType> &&
    requires(
        Sys sys, typename Sys::ScalarType t, typename Sys::ScalarType* x,
        typename Sys::ScalarType* u, typename Sys::ScalarType* dxdt) {
      typename Sys::ScalarType;
      { sys.get_n_x() } -> std::convertible_to<int>;
      { sys.get_n_u() } -> std::convertible_to<int>;
      { sys.dynamics(t, x, u, dxdt) };
    };

template <typename Sys>
concept BoundedSystem = DynamicalSystem<Sys> && requires(Sys sys) {
  {
    sys.get_x_lower_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_x_upper_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_u_lower_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_u_upper_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
};

template <typename I, typename Sys>
concept Integrator =
    DynamicalSystem<Sys> &&
    requires(
        I integrator, Sys sys, typename Sys::ScalarType dt,
        typename Sys::ScalarType t, typename Sys::ScalarType* x,
        typename Sys::ScalarType* u, typename Sys::ScalarType* x_next) {
      { integrator(sys, dt, t, x, u, x_next) };
    };

// A thing that does Rollouts given an x0 and inputs. Effectively a "Plant"
template <typename P, typename Scalar>
concept RolloutProvider = requires(
    P plant, TensorView<Scalar, 1> x0, TensorView<Scalar, 3> u_seq, Scalar dt) {
  { plant(x0, u_seq, dt) } -> std::convertible_to<TensorView<Scalar, 3>>;
};

// One constant timestep per candidate, rather than per integration step.
template <typename P, typename Scalar>
concept VariableTimeRolloutProvider =
    requires(P plant, TensorView<Scalar, 1> x0, TensorView<Scalar, 3> u_seq,
             TensorView<Scalar, 1> dts) {
      { plant(x0, u_seq, dts) } -> std::convertible_to<TensorView<Scalar, 3>>;
    };

// template <typename P>
// concept Policy = requires(P policy, float t, ) {
//   { policy(t, u) };
// };

// NOTE(amg): RK2 can allocate it's own memory as needed, but in fact in this
// case, does not even need to be a functor at all
class RK2
{
 public:
  template <DynamicalSystem Sys>
  __host__ __device__ void operator()(
      Sys& sys, typename Sys::ScalarType dt, typename Sys::ScalarType t,
      const typename Sys::ScalarType* x, const typename Sys::ScalarType* u,
      typename Sys::ScalarType* x_next)
  {
    typename Sys::ScalarType k1[64];
    typename Sys::ScalarType k2[64];

    sys.dynamics(t, x, u, k1);
    for (int i = 0; i < sys.get_n_x(); ++i) x_next[i] = x[i] + dt * k1[i];
    sys.dynamics(t, x_next, u, k2);
    for (int i = 0; i < sys.get_n_x(); ++i)
      x_next[i] = x[i] + dt * 0.5f * (k1[i] + k2[i]);
  }
};

class Euler
{
 public:
  template <DynamicalSystem Sys>
  __host__ __device__ void operator()(
      Sys& sys, typename Sys::ScalarType dt, typename Sys::ScalarType t,
      const typename Sys::ScalarType* x, const typename Sys::ScalarType* u,
      typename Sys::ScalarType* x_next)
  {
    typename Sys::ScalarType k1[64];

    sys.dynamics(t, x, u, k1);
    for (int i = 0; i < sys.get_n_x(); ++i) x_next[i] = x[i] + dt * k1[i];
  }
};

template <typename Scalar>
__host__ __device__ void clamp_to_bounds(
    const Scalar* lb, const Scalar* ub, int n_u, Scalar* u)
{
  for (int i = 0; i < n_u; ++i) u[i] = fmaxf(fminf(u[i], ub[i]), lb[i]);
}

template <DynamicalSystem Sys, typename Integrator>
__host__ __device__ void rollout(
    int batch_idx, Sys& sys, Integrator& integrator,
    const TensorView<typename Sys::ScalarType, 1>& x0,
    TensorView<typename Sys::ScalarType, 3> u_seq, typename Sys::ScalarType dt,
    TensorView<typename Sys::ScalarType, 3> x_seq)
{
  int T = u_seq.shape(1);
  x_seq.template slice_1d<2>(batch_idx, 0).deep_copy_from(x0);
  float t = 0.0;
  for (int i = 0; i < T; ++i)
  {
    if constexpr (BoundedSystem<Sys>)
    {
      clamp_to_bounds(
          sys.get_u_lower_bounds(), sys.get_u_upper_bounds(), sys.get_n_u(),
          u_seq.template slice_1d<2>(batch_idx, i).data());
    }
    integrator(
        sys, dt, t, x_seq.template slice_1d<2>(batch_idx, i),
        u_seq.template slice_1d<2>(batch_idx, i),
        x_seq.template slice_1d<2>(batch_idx, i + 1));

    if constexpr (BoundedSystem<Sys>)
    {
      clamp_to_bounds(
          sys.get_x_lower_bounds(), sys.get_x_upper_bounds(), sys.get_n_x(),
          x_seq.template slice_1d<2>(batch_idx, i + 1).data());
    }
    t += dt;
  }
}

template <DynamicalSystem Sys, typename Integrator>
__global__ void rollout_kernel(
    Sys sys, Integrator integrator, const TensorView<float, 1> x0,
    TensorView<float, 3> u_seq, float dt, TensorView<float, 3> x_seq)
{
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_size = u_seq.shape(0);
  if (batch_idx >= batch_size) return;
  rollout(batch_idx, sys, integrator, x0, u_seq, dt, x_seq);
}

template <DynamicalSystem Sys, typename Integrator>
void rollout_gpu(
    TensorView<typename Sys::ScalarType, 3> x_seq, const Sys& sys,
    const Integrator& integrator, const TensorView<float, 1>& x0,
    TensorView<float, 3>& u_seq, float dt,
    std::optional<std::tuple<int, int>> grid_block = std::nullopt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);

  int grid = (batch_size + 255) / 256;
  int block = 256;
  if (grid_block.has_value())
  {
    grid = std::get<0>(grid_block.value());
    block = std::get<1>(grid_block.value());
  }
  rollout_kernel<<<grid, block>>>(sys, integrator, x0, u_seq, dt, x_seq);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Note u is non-const due to bounds clamping
// TODO(amg), maybe this thing need not to take a system at all, but rather a
// function pointer to integrate and a struct with n_x, n_u, and bounds.
template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_gpu(
    const Sys& sys, const Integrator& integrator,
    const TensorView<float, 1>& x0, TensorView<float, 3>& u_seq, float dt,
    std::optional<std::tuple<int, int>> grid_block = std::nullopt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);
  int n_x = sys.get_n_x();
  Tensor<float, 3> x_seq(batch_size, T + 1, n_x);
  rollout_gpu(x_seq.view(), sys, integrator, x0, u_seq, dt, grid_block);
  return x_seq;
}

template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_cpu(
    Sys& sys, Integrator& integrator, const TensorView<double, 1>& x0,
    TensorView<double, 3>& u_seq, float dt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);
  int n_x = sys.get_n_x();
  Tensor<double, 3> x_seq(batch_size, T + 1, n_x);
  for (int b = 0; b < batch_size; ++b)
    rollout(b, sys, integrator, x0, u_seq, dt, x_seq.view());
  return x_seq;
}

template <DynamicalSystem Sys, typename Integrator>
class PlantCPU
{
 public:
  PlantCPU(Sys& sys, Integrator& integrator) : sys(sys), integrator(integrator)
  {
  }

  Tensor<typename Sys::ScalarType, 3> operator()(
      const TensorView<typename Sys::ScalarType, 1>& x0,
      TensorView<typename Sys::ScalarType, 3> u_seq, float dt)
  {
    return rollout_cpu(sys, integrator, x0, u_seq, dt);
  }

 private:
  Sys& sys;
  Integrator& integrator;
};

// Timestep lists must be host-readable for validation; GPU rollouts use
// managed memory so the same list is also accessible inside the kernel.
inline void validate_rollout_dts(const TensorView<float, 1>& dts,
                                 int batch_size)
{
  if (dts.shape(0) != batch_size || dts.memory == Memory::Device)
    throw std::invalid_argument("dts must be host-readable, one per rollout");
  for (int b = 0; b < batch_size; ++b)
    if (!std::isfinite(dts(b)) || dts(b) <= 0.0f)
      throw std::invalid_argument(
          "rollout timesteps must be finite and positive");
}

template <DynamicalSystem Sys, typename Integrator>
__global__ void rollout_variable_time_kernel(Sys sys, Integrator integrator,
                                             TensorView<float, 1> x0,
                                             TensorView<float, 3> u_seq,
                                             TensorView<float, 1> dts,
                                             TensorView<float, 3> x_seq)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < u_seq.shape(0))
    rollout(b, sys, integrator, x0, u_seq, dts(b), x_seq);
}

template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_gpu(
    const Sys& sys, const Integrator& integrator,
    const TensorView<float, 1>& x0, TensorView<float, 3>& u_seq,
    const TensorView<float, 1>& dts,
    std::optional<std::tuple<int, int>> grid_block = std::nullopt)
{
  validate_rollout_dts(dts, u_seq.shape(0));
  if (dts.memory != Memory::Managed)
    throw std::invalid_argument("GPU timestep lists must use managed memory");
  Tensor<float, 3> x_seq(u_seq.shape(0), u_seq.shape(1) + 1, sys.get_n_x());
  int grid = (u_seq.shape(0) + 255) / 256;
  int block = 256;
  if (grid_block)
  {
    grid = std::get<0>(*grid_block);
    block = std::get<1>(*grid_block);
  }
  rollout_variable_time_kernel<<<grid, block>>>(sys, integrator, x0, u_seq,
                                                dts, x_seq.view());
  CUDA_CHECK(cudaDeviceSynchronize());
  return x_seq;
}

template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_cpu(
    Sys& sys, Integrator& integrator, const TensorView<double, 1>& x0,
    TensorView<double, 3>& u_seq, const TensorView<float, 1>& dts)
{
  validate_rollout_dts(dts, u_seq.shape(0));
  Tensor<double, 3> x_seq(u_seq.shape(0), u_seq.shape(1) + 1, sys.get_n_x());
  for (int b = 0; b < u_seq.shape(0); ++b)
    rollout(b, sys, integrator, x0, u_seq, static_cast<double>(dts(b)),
            x_seq.view());
  return x_seq;
}

}  // namespace sds