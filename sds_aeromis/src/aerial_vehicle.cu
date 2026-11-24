#include <iostream>

#include "aeromis_core/util/array.hpp"
#include "sds_aeromis/aerial_vehicle.cuh"

namespace sds_aeromis
{

// ---------------------------------------------------------------------------
// AerialVehicle
// ---------------------------------------------------------------------------

template <typename Mem, typename Scalar>
__host__ AerialVehicle<Mem, Scalar>::AerialVehicle(
    aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle,
    aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu,
    uint8_t* memory_buffer)
    : aerial_vehicle_(aerial_vehicle), aerial_vehicle_cpu_(aerial_vehicle_cpu),
      memory_buffer(memory_buffer)
{
  n_x = aerial_vehicle_cpu->get_n_x();
  n_u = aerial_vehicle_cpu->get_n_u();
}

template <typename Mem, typename Scalar>
__device__ __host__ void AerialVehicle<Mem, Scalar>::dynamics(
    Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt)
{
#ifdef __CUDA_ARCH__
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
#else
  int batch_idx = 0;
#endif
  aeromis_core::Array<Mem, Scalar> x_view(const_cast<Scalar*>(x), n_x);
  aeromis_core::Array<Mem, Scalar> u_view(const_cast<Scalar*>(u), n_u);
  aeromis_core::Array<Mem, Scalar> dxdt_view(dxdt, n_x);
  aerial_vehicle_->dynamics(
      t, x_view, u_view, dxdt_view, batch_idx,
      memory_buffer + batch_idx * aerial_vehicle_->required_bytes_per_thread());
}

template <typename Mem, typename Scalar>
__host__ bool AerialVehicle<Mem, Scalar>::thrust_is_delayed(int i) const
{
  return aerial_vehicle_cpu_->get_description()->thrust_sources[i].delay > 0;
}

template <typename Mem, typename Scalar>
__host__ __device__ void AerialVehicle<Mem, Scalar>::advance_delayed_actuators(
    Scalar h, const Scalar* x, const Scalar* u, Scalar* out) const
{
#ifdef __CUDA_ARCH__
  aerial_vehicle_->advance_delayed_actuators(h, x, u, out);
#else
  aerial_vehicle_cpu_->advance_delayed_actuators(h, x, u, out);
#endif
}

template <typename Mem, typename Scalar>
__host__ __device__ int AerialVehicle<Mem, Scalar>::get_n_x() const
{
  return n_x;
}

template <typename Mem, typename Scalar>
__host__ __device__ int AerialVehicle<Mem, Scalar>::get_n_u() const
{
  return n_u;
}

template <typename Mem, typename Scalar>
__host__ std::vector<std::string> AerialVehicle<Mem, Scalar>::get_x_names()
    const
{
  return aerial_vehicle_cpu_->get_state_parameters().get_names();
}

template <typename Mem, typename Scalar>
__host__ std::vector<std::string> AerialVehicle<Mem, Scalar>::get_joint_names()
    const
{
  return aerial_vehicle_cpu_->get_joint_names();
}

template <typename Mem, typename Scalar>
__host__ std::vector<std::string> AerialVehicle<Mem, Scalar>::get_u_names()
    const
{
  return aerial_vehicle_cpu_->get_input_parameters().get_names();
}

template <typename Mem, typename Scalar>
__host__ __device__ Scalar* AerialVehicle<Mem, Scalar>::get_x_upper_bounds()
    const
{
#ifdef __CUDA_ARCH__
  auto* ptr = aerial_vehicle_;
#else
  auto* ptr = aerial_vehicle_cpu_;
#endif
  return ptr->get_state_parameters_ref().get_upper_bounds_ref().get_data();
}

template <typename Mem, typename Scalar>
__host__ __device__ Scalar* AerialVehicle<Mem, Scalar>::get_x_lower_bounds()
    const
{
#ifdef __CUDA_ARCH__
  auto* ptr = aerial_vehicle_;
#else
  auto* ptr = aerial_vehicle_cpu_;
#endif
  return ptr->get_state_parameters_ref().get_lower_bounds_ref().get_data();
}

template <typename Mem, typename Scalar>
__host__ __device__ Scalar* AerialVehicle<Mem, Scalar>::get_u_upper_bounds()
    const
{
#ifdef __CUDA_ARCH__
  auto* ptr = aerial_vehicle_;
#else
  auto* ptr = aerial_vehicle_cpu_;
#endif
  return ptr->get_input_parameters_ref().get_upper_bounds_ref().get_data();
}

template <typename Mem, typename Scalar>
__host__ __device__ Scalar* AerialVehicle<Mem, Scalar>::get_u_lower_bounds()
    const
{
#ifdef __CUDA_ARCH__
  auto* ptr = aerial_vehicle_;
#else
  auto* ptr = aerial_vehicle_cpu_;
#endif
  return ptr->get_input_parameters_ref().get_lower_bounds_ref().get_data();
}

// Explicit instantiations — the only two supported configurations.
template class AerialVehicle<aeromis_core::GPU, float>;
template class AerialVehicle<aeromis_core::CPU, double>;

// ---------------------------------------------------------------------------
// AerialVehicleManager
// ---------------------------------------------------------------------------

__host__ AerialVehicleManager::AerialVehicleManager(
    const std::filesystem::path& description_path,
    std::optional<std::string> flow_field_path)
{
  aeromis_core::AerialVehicleDescription<aeromis_core::GPU, float> description(
      description_path);
  aeromis_core::AerialVehicleDescription<aeromis_core::CPU, double>
      description_cpu(description_path);

  aerial_vehicle_gpu_ =
      std::make_unique<aeromis_core::AerialVehicle<aeromis_core::GPU, float>>(
          std::move(description), flow_field_path);
  aerial_vehicle_gpu_->copy_members_to_gpu(
      aerial_vehicle_gpu_->get_gpu_instance());
  cudaMallocManaged(
      &gpu_memory_buffer,
      aerial_vehicle_gpu_->required_bytes_per_thread() * 1024);

  aerial_vehicle_cpu_ =
      std::make_unique<aeromis_core::AerialVehicle<aeromis_core::CPU, double>>(
          std::move(description_cpu), flow_field_path);
  cpu_memory_buffer.resize(
      aerial_vehicle_cpu_->required_bytes_per_thread() * 1024);
}

AerialVehicleManager::AerialVehicleManager(AerialVehicleManager&& other)
{
  aerial_vehicle_gpu_ = std::move(other.aerial_vehicle_gpu_);
  aerial_vehicle_cpu_ = std::move(other.aerial_vehicle_cpu_);
  gpu_memory_buffer = other.gpu_memory_buffer;
  cpu_memory_buffer = std::move(other.cpu_memory_buffer);
  other.gpu_memory_buffer = nullptr;
}

__device__ __host__ AerialVehicleManager::~AerialVehicleManager()
{
  cudaFree(gpu_memory_buffer);
}

__host__ AerialVehicle<aeromis_core::GPU, float>
AerialVehicleManager::get_aerial_vehicle_gpu()
{
  return AerialVehicle<aeromis_core::GPU, float>(
      aerial_vehicle_gpu_->get_gpu_instance(), aerial_vehicle_gpu_.get(),
      gpu_memory_buffer);
}

__host__ AerialVehicle<aeromis_core::CPU, double>
AerialVehicleManager::get_aerial_vehicle_cpu()
{
  return AerialVehicle<aeromis_core::CPU, double>(
      aerial_vehicle_cpu_.get(), aerial_vehicle_cpu_.get(),
      cpu_memory_buffer.data());
}

template <typename Mem, typename Scalar>
__host__ void AerialVehicle<Mem, Scalar>::get_dfdx(
    Scalar t, const float* x, const float* u, float* dfdx)
{
  std::vector<float> dfdx_vec(n_x * n_x);
  std::vector<Scalar> u_vec(u, u + n_u);
  double delta = 1e-6;
  for (int i = 0; i < n_x; ++i)
  {
    std::vector<Scalar> x_plus(x, x + n_x);
    std::vector<Scalar> x_minus(x, x + n_x);
    x_plus[i] += delta;
    x_minus[i] -= delta;

    std::vector<Scalar> dxdt_plus(n_x);
    std::vector<Scalar> dxdt_minus(n_x);
    dynamics(t, x_plus.data(), u_vec.data(), dxdt_plus.data());
    dynamics(t, x_minus.data(), u_vec.data(), dxdt_minus.data());

    for (int j = 0; j < n_x; ++j)
      dfdx_vec[j * n_x + i] = (dxdt_plus[j] - dxdt_minus[j]) / (2 * delta);
  }
  std::copy(dfdx_vec.begin(), dfdx_vec.end(), dfdx);
}

template <typename Mem, typename Scalar>
__host__ void AerialVehicle<Mem, Scalar>::get_dfdu(
    Scalar t, const float* x, const float* u, float* dfdu)
{
  std::vector<float> dfdu_vec(n_x * n_u);
  std::vector<Scalar> x_vec(x, x + n_x);
  double delta = 1e-6;
  for (int i = 0; i < n_u; ++i)
  {
    std::vector<Scalar> u_plus(u, u + n_u);
    std::vector<Scalar> u_minus(u, u + n_u);
    u_plus[i] += delta;
    u_minus[i] -= delta;

    std::vector<Scalar> dxdt_plus(n_x);
    std::vector<Scalar> dxdt_minus(n_x);
    dynamics(t, x_vec.data(), u_plus.data(), dxdt_plus.data());
    dynamics(t, x_vec.data(), u_minus.data(), dxdt_minus.data());

    for (int j = 0; j < n_x; ++j)
      dfdu_vec[j * n_u + i] = (dxdt_plus[j] - dxdt_minus[j]) / (2 * delta);
  }
  std::copy(dfdu_vec.begin(), dfdu_vec.end(), dfdu);
}

// ---------------------------------------------------------------------------
// rollout_to_csv
// ---------------------------------------------------------------------------

void rollout_to_csv(
    AerialVehicleManager& manager, const TensorView<float, 2>& x_seq,
    const TensorView<float, 2>& u_seq, float dt)
{
  for (auto name : manager.get_aerial_vehicle_gpu().get_x_names())
    std::cout << name << ",";
  for (auto name : manager.get_aerial_vehicle_gpu().get_u_names())
    std::cout << name << ",";
  std::cout << std::endl;

  for (int t = 0; t < x_seq.shape(0); ++t)
  {
    for (int i = 0; i < x_seq.shape(1); ++i) std::cout << x_seq(t, i) << ",";
    for (int i = 0; i < u_seq.shape(1); ++i)
      std::cout << u_seq(std::min(t, u_seq.shape(0) - 1), i) << ",";
    std::cout << std::endl;
  }
}

}  // namespace sds_aeromis
