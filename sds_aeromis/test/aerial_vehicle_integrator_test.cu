#include <unistd.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unsupported/Eigen/MatrixFunctions>

#include "sds_aeromis/aerial_vehicle.cuh"
#include "sds_aeromis/aerial_vehicle_integrator.cuh"

using namespace aeromis_core;

namespace
{
void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message)
{
  require(
      std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
      message);
}

// Independent reference: augmented matrix exponential, including the input.
template <typename Scalar>
void exact_source_test()
{
  for (double tau : {0.001, 0.05, 1.0})
    for (double a : {0.0, -1e-10, -17.0, -3.0 / tau, -2000.0})
      for (double h : {0.0, 1e-9, 0.01, 0.05, 0.08, 0.5})
      {
        ThrustSource<Scalar> source;
        source.delay = tau;
        source.a = a;
        source.b = 86.5;
        tau = double(source.delay);
        a = double(source.a);
        Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
        M(0, 0) = a;
        M(0, 2) = -source.b;
        M(0, 3) = source.b;
        M(1, 2) = 1 / tau;
        M(2, 1) = -12 / tau;
        M(2, 2) = -6 / tau;
        M(2, 3) = 12 / tau;
        Eigen::Vector4d initial(0.7, 0.2, -0.1, 0.6);
        Eigen::Vector4d expected = (h * M).exp() * initial;
        Scalar z[2] = {Scalar(initial[1]), Scalar(initial[2])}, thrust;
        const double tolerance = std::is_same_v<Scalar, float> ? 3e-5 : 2e-9;
        source.advance_delay(h, initial[3], initial[0], z, thrust, z);
        near(
            thrust, expected[0], tolerance * (1 + std::abs(expected[0])),
            "exact thrust");
        near(z[0], expected[1], tolerance, "exact z1");
        near(z[1], expected[2], tolerance, "exact z2");
        // Steady initial conditions must remain steady at every coarse dt.
        if (a < -1e-5)
        {
          Scalar eq[2] = {Scalar(0.6), 0};
          source.advance_delay(h, 0.6, -source.b * 0.6 / a, eq, thrust, eq);
          near(
              thrust, -source.b * 0.6 / a,
              tolerance * (1 + std::abs(source.b * 0.6 / a)), "steady thrust");
          near(eq[0], 0.6, tolerance, "steady z1");
          near(eq[1], 0, tolerance, "steady z2");
        }
      }
}

struct VehicleFile
{
  std::filesystem::path path;

  explicit VehicleFile(const std::vector<std::string>& delays)
  {
    std::string pattern = "/tmp/aeromis_exact_XXXXXX";
    int fd = mkstemp(pattern.data());
    if (fd == -1) throw std::runtime_error("Could not create test model");
    close(fd);
    path = pattern;
    std::ofstream f(path);
    f << R"(<robot name="test"><model><options><cg_pos xyz="0 0 0"/>
      <body_drag area="0" coefficient="0"/></options>
      <surface name="surface"><parent link="body"/>
      <origin xyz="0 0 0" rpy="0 0 0"/><force area="0" coefficient="0"/>
      <backwash use="false"/></surface>)";
    for (size_t i = 0; i < delays.size(); ++i)
    {
      f << "<thrust_source name=\"prop" << i << "\">" << R"(
        <parent link="body"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <prop_diameter value="0.1"/><torque_coefficient value="0"/>
        <thrust_dynamics_a value="-17"/><thrust_dynamics_b value="86.5"/>
        <thrust_offset value="0"/><thrust_limit lower="0" upper="10"/>
        <throttle_limit lower="0" upper="1"/>)";
      if (!delays[i].empty())
        f << "<thrust_delay_ms value=\"" << delays[i] << "\"/>";
      f << "</thrust_source>";
    }
    f << R"(</model><link name="body"><inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1"/></inertial></link><link name="child"/>
      <joint name="joint" type="revolute"><parent link="body"/>
      <child link="child"/><axis xyz="0 1 0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <limit lower="-1" upper="1" velocity="2"/></joint></robot>)";
  }

  ~VehicleFile()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

void vehicle_test(const std::vector<std::string>& delays)
{
  VehicleFile file(delays);
  AerialVehicleDescription<CPU, double> desc(file.path);
  AerialVehicle<CPU, double> core(std::move(desc));
  std::vector<uint8_t> scratch(core.required_bytes_per_thread());
  sds_aeromis::AerialVehicle<CPU, double> sys(&core, &core, scratch.data());
  const int nx = sys.get_n_x(), nu = sys.get_n_u();
  std::vector<double> x(nx, 0), output(nx), legacy(nx), alias(nx), u(nu, 0.4);
  x[3] = 1;
  u[0] = 0;
  sds_aeromis::AerialVehicleIntegrator integrator;
  if (!sys.has_thrust_delay())
  {
    integrator(sys, 0.08, 0., x.data(), u.data(), output.data());
    sds::RK2{}(sys, 0.08, 0., x.data(), u.data(), legacy.data());
    require(output == legacy, "undelayed URDF must preserve legacy RK2");
    return;
  }
  for (double h : {0.01, 0.05, 0.08})
  {
    std::fill(x.begin(), x.end(), 0);
    x[3] = 1;
    // Many coarse steps must remain finite, and in-place estimates must agree.
    for (int step = 0; step < 150; ++step)
    {
      alias = x;
      integrator(sys, h, step * h, x.data(), u.data(), output.data());
      integrator(sys, h, step * h, alias.data(), u.data(), alias.data());
      for (int j = 0; j < nx; ++j)
      {
        require(std::isfinite(output[j]), "finite coarse rollout");
        near(alias[j], output[j], 1e-12, "in-place integration");
      }
      // Match the controller's estimate against the simulator actuator step.
      alias = x;
      sys.advance_delayed_actuators(h, alias.data(), u.data(), alias.data());
      int pade = 14 + delays.size();
      for (int i = 0; i < int(delays.size()); ++i)
      {
        if (!sys.thrust_is_delayed(i)) continue;
        near(
            alias[14 + i], output[14 + i], 1e-12, "controller thrust estimate");
        near(alias[pade], output[pade], 1e-12, "controller z1 estimate");
        near(
            alias[pade + 1], output[pade + 1], 1e-12, "controller z2 estimate");
        pade += 2;
      }
      // The body endpoint stage must see exact delayed thrust, not an Euler
      // prediction of that fast subsystem. This fixture has only axial force.
      double expected_velocity = x[7];
      const auto* description = core.get_description();
      for (int i = 0; i < int(delays.size()); ++i)
      {
        const auto& motor = description->thrust_sources[i];
        const double stage_thrust =
            sys.thrust_is_delayed(i)
                ? alias[14 + i]
                : x[14 + i] + h * motor.dynamics(x[14 + i], u[1 + i]);
        const auto bounded = [&](double value)
        {
          return std::min(
              std::max(value, double(motor.min_thrust)),
              double(motor.max_thrust));
        };
        expected_velocity +=
            0.5 * h * (bounded(x[14 + i]) + bounded(stage_thrust));
      }
      near(output[7], expected_velocity, 1e-10, "exact endpoint force stage");
      x = output;
    }
    // 50/80 ms steps are numerically stable for the delayed actuator block.
    for (int i = 0; i < int(delays.size()); ++i)
      if (sys.thrust_is_delayed(i))
        near(x[14 + i], 86.5 * 0.4 / 17, 1e-8, "coarse steady thrust");
  }
}
}  // namespace

int main()
{
  try
  {
    exact_source_test<double>();
    exact_source_test<float>();
    vehicle_test({"", "0", ""});
    vehicle_test({"50", "0", "100"});
    vehicle_test({"50", "50", "50"});
    std::cout
        << "Exact-source, equilibrium, coarse-rollout, mixed/absent delay, "
           "aliasing, and controller-estimate checks passed.\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
