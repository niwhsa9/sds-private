#pragma once

#include <aeromis_core/dynamical_systems/aerial_vehicle.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "sds_control.cuh"
#include "sds_core.cuh"

namespace sds_aeromis
{

template <typename Mem, typename Scalar>
class AerialVehicle
{
 public:
  using ScalarType = Scalar;

  __host__ AerialVehicle(
      aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle,
      aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu,
      uint8_t* memory_buffer);

  __device__ __host__ void dynamics(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt);

  __host__ __device__ bool has_thrust_delay() const
  {
    return n_x > 13 + n_u;
  }

  // Host queries are used to preserve legacy estimates for undelayed motors.
  __host__ bool thrust_is_delayed(int source_index) const;

  __host__ __device__ void advance_delayed_actuators(
      Scalar h, const Scalar* x, const Scalar* u, Scalar* out) const;

  __host__ __device__ int get_n_x() const;
  __host__ __device__ int get_n_u() const;

  //__host__ void get_dfdx(
  //    Scalar t, const Scalar* x, const Scalar* u, Scalar* dfdx);
  __host__ void get_dfdx(Scalar t, const float* x, const float* u, float* dfdx);
  __host__ void get_dfdu(Scalar t, const float* x, const float* u, float* dfdu);

  //__host__
  // void get_dfdu(Scalar t, const Scalar* x, const Scalar* u, Scalar* dfdx);

  __host__ std::vector<std::string> get_x_names() const;
  __host__ std::vector<std::string> get_u_names() const;
  __host__ std::vector<std::string> get_joint_names() const;

  __host__ __device__ Scalar* get_x_upper_bounds() const;
  __host__ __device__ Scalar* get_x_lower_bounds() const;
  __host__ __device__ Scalar* get_u_upper_bounds() const;
  __host__ __device__ Scalar* get_u_lower_bounds() const;

 private:
  aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_;
  aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu_;
  uint8_t* memory_buffer;
  int n_x, n_u;
};

// Extern declarations for the only two supported instantiations.
extern template class AerialVehicle<aeromis_core::GPU, float>;
extern template class AerialVehicle<aeromis_core::CPU, double>;

class AerialVehicleManager
{
 public:
  __host__ explicit AerialVehicleManager(
      const std::filesystem::path& description_path);

  __device__ __host__ ~AerialVehicleManager();

  AerialVehicleManager(AerialVehicleManager&& other);

  __host__ AerialVehicle<aeromis_core::GPU, float> get_aerial_vehicle_gpu();
  __host__ AerialVehicle<aeromis_core::CPU, double> get_aerial_vehicle_cpu();

 private:
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::CPU, double>>
      aerial_vehicle_cpu_;
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::GPU, float>>
      aerial_vehicle_gpu_;
  uint8_t* gpu_memory_buffer;
  std::vector<uint8_t> cpu_memory_buffer;
};

void rollout_to_csv(
    AerialVehicleManager& manager, const TensorView<float, 2>& x_seq,
    const TensorView<float, 2>& u_seq, float dt);

}  // namespace sds_aeromis
