#pragma once

#include "sds_core.cuh"

namespace sds_aeromis
{
// Heun RK2 for the vehicle, exact held-throttle evolution for delayed motors.
// Both the endpoint force evaluation and final actuator states use the exact
// coupled thrust/Pade solution. Undelayed URDFs follow the original RK2 path.
class AerialVehicleIntegrator
{
 public:
  template <typename Sys>
  __host__ __device__ void operator()(
      Sys& sys, typename Sys::ScalarType h, typename Sys::ScalarType t,
      const typename Sys::ScalarType* x, const typename Sys::ScalarType* u,
      typename Sys::ScalarType* out) const
  {
    if (!sys.has_thrust_delay())
    {
      sds::RK2{}(sys, h, t, x, u, out);
      return;
    }
    using Scalar = typename Sys::ScalarType;
    // Match SDS's existing 64-state integrator capacity. Keep stage storage
    // separate: x and out may alias, and exact updates need the original x.
    Scalar k1[64], stage[64];
    sys.dynamics(t, x, u, k1);
    for (int i = 0; i < sys.get_n_x(); ++i) stage[i] = x[i] + h * k1[i];
    sys.advance_delayed_actuators(h, x, u, stage);
    Scalar k2[64];
    sys.dynamics(t + h, stage, u, k2);
    for (int i = 0; i < sys.get_n_x(); ++i)
      stage[i] = x[i] + h * Scalar(0.5) * (k1[i] + k2[i]);
    sys.advance_delayed_actuators(h, x, u, stage);
    for (int i = 0; i < sys.get_n_x(); ++i) out[i] = stage[i];
  }
};
}  // namespace sds_aeromis
