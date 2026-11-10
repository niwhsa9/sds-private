#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "tensor.cuh"

namespace sds
{
// Stationary discrete Riccati solution. Continuous Jacobians are converted
// with forward Euler when discretize=true. In both modes Q and R are cost
// rates multiplied by dt, matching compute_tvlqr_gains.
// K is nX x nU state-major, with the negative feedback sign already included.
std::pair<Tensor<float, 2>, Tensor<float, 2>>
compute_infinite_horizon_lqr_gains(const TensorView<float, 2>& A,
                                   const TensorView<float, 2>& B,
                                   const TensorView<float, 2>& Q,
                                   const TensorView<float, 2>& R, float dt,
                                   bool discretize = true,
                                   double tolerance = 1e-10,
                                   int max_iterations = 10000);

// Constant affine feedback. References must describe an equilibrium of the
// model used to design K if asymptotic regulation is required. This class
// deliberately imposes no quaternion or application-specific state convention.
struct InfiniteHorizonPolicy
{
  Tensor<float, 2> K, P;
  Tensor<float, 1> x_reference, u_reference;
  std::optional<Tensor<float, 1>> u_lower, u_upper;

  InfiniteHorizonPolicy(Tensor<float, 2>&& K_, Tensor<float, 2>&& P_,
                        Tensor<float, 1>&& x_reference_,
                        Tensor<float, 1>&& u_reference_)
      : K(std::move(K_)),
        P(std::move(P_)),
        x_reference(std::move(x_reference_)),
        u_reference(std::move(u_reference_))
  {
    const int nx = x_reference.shape(0), nu = u_reference.shape(0);
    if (nx <= 0 || nu <= 0 || K.shape(0) != nx || K.shape(1) != nu ||
        P.shape(0) != nx || P.shape(1) != nx)
      throw std::invalid_argument("Infinite-horizon policy dimension mismatch");
  }

  InfiniteHorizonPolicy(const InfiniteHorizonPolicy& other)
      : K(other.K.clone()),
        P(other.P.clone()),
        x_reference(other.x_reference.clone()),
        u_reference(other.u_reference.clone())
  {
    if (other.u_lower) u_lower = other.u_lower->clone();
    if (other.u_upper) u_upper = other.u_upper->clone();
  }
  InfiniteHorizonPolicy(InfiniteHorizonPolicy&&) = default;
  InfiniteHorizonPolicy& operator=(InfiniteHorizonPolicy&&) = default;
  InfiniteHorizonPolicy& operator=(const InfiniteHorizonPolicy& other)
  {
    if (this != &other) *this = InfiniteHorizonPolicy(other);
    return *this;
  }

  Tensor<float, 1> eval_policy(const float* x) const
  {
    const int nx = x_reference.shape(0), nu = u_reference.shape(0);
    if ((u_lower && u_lower->shape(0) != nu) ||
        (u_upper && u_upper->shape(0) != nu))
      throw std::invalid_argument(
          "Infinite-horizon input bound dimension mismatch");
    Tensor<float, 1> u(Memory::Host, nu);
    for (int j = 0; j < nu; ++j)
    {
      double value = u_reference(j);
      for (int i = 0; i < nx; ++i)
      {
        // Ignore unobserved states, including application-specific manifolds.
        if (K(i, j) != 0.0f) value += K(i, j) * double(x[i] - x_reference(i));
      }
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite infinite-horizon command");
      if (u_lower && u_upper && (*u_lower)(j) > (*u_upper)(j))
        throw std::invalid_argument("Inverted infinite-horizon input bounds");
      if (u_lower) value = std::max(value, double((*u_lower)(j)));
      if (u_upper) value = std::min(value, double((*u_upper)(j)));
      u(j) = static_cast<float>(value);
    }
    return u;
  }
};
}  // namespace sds
