#include "sds_mpc.cuh"

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace std::chrono_literals;

void check(bool value, const char* message)
{
  if (!value) throw std::runtime_error(message);
}

struct Plant
{
  std::shared_ptr<std::atomic<bool>> destroyed;
  explicit Plant(std::shared_ptr<std::atomic<bool>> flag = {}) : destroyed(flag) {}
  Plant(Plant&&) = default;
  ~Plant() { if (destroyed) destroyed->store(true); }
  Tensor<float, 3> operator()(
      TensorView<float, 1> x, TensorView<float, 3> u, float)
  {
    Tensor<float, 3> result(Memory::Host, u.shape(0), u.shape(1) + 1, x.shape(0));
    for (int b = 0; b < u.shape(0); ++b)
      for (int k = 0; k <= u.shape(1); ++k)
        for (int i = 0; i < x.shape(0); ++i) result(b, k, i) = x(i);
    return result;
  }
};

struct Gate
{
  std::promise<void> entered, release;
  std::shared_future<void> released = release.get_future().share();
};

struct Optimizer
{
  std::shared_ptr<Gate> gate;
  std::atomic<int> calls = 0;
  sds::LinearPolicy operator()(TensorView<float, 1>, double, sds::LinearPolicy* warm)
  {
    ++calls;
    if (gate)
    {
      gate->entered.set_value();
      gate->released.wait();
    }
    return *warm;
  }
};
using Manager = sds::ReplanManager<Plant, Optimizer>;

sds::LinearPolicy make_policy()
{
  Tensor<float, 3> K(2, 13, 1), P(3, 13, 13);
  Tensor<float, 2> x(3, 13), u(2, 1);
  K.fill(0); P.fill(0); x.fill(0); u.fill(0);
  for (int k = 0; k < 3; ++k) x(k, 3) = 1.;
  return sds::LinearPolicy(std::move(K), std::move(P), std::move(x), std::move(u), 0., .01);
}

int main()
{
  // This regression needs no GPU: the old detached worker hangs on CV teardown.
  for (int i = 0; i < 32; ++i)
  {
    auto destroyed = std::make_shared<std::atomic<bool>>(false);
    {
      Manager manager(Plant{destroyed}, std::make_shared<Optimizer>());
      check(manager.ready_to_replan(), "Manager starts ready");
      std::this_thread::sleep_for(2ms);  // Let the worker enter its idle wait.
    }
    check(destroyed->load(), "Idle destruction releases the plant");
  }
  std::cout << "Idle worker shutdown passed" << std::endl;

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
  {
    std::cout << "Skipping solve tests: managed-memory tensors require a CUDA device\n";
    return 0;
  }
  auto policy = make_policy();
  Tensor<float, 1> x(13);
  x.fill(0); x(3) = 1.;
  auto optimizer = std::make_shared<Optimizer>();
  {
    Manager manager(Plant{}, optimizer);
    manager.request_replan(2, 1, 0., x.view(), policy);
    auto result = manager.get_policy_blocking();
    check(result && result->nX == 13 && optimizer->calls == 1,
          "Normal replanning still returns its policy");
  }

  // Hold an actual solve open and verify teardown preserves its resources.
  auto destroyed = std::make_shared<std::atomic<bool>>(false);
  optimizer->gate = std::make_shared<Gate>();
  auto entered = optimizer->gate->entered.get_future();
  auto manager = std::make_unique<Manager>(Plant{destroyed}, optimizer);
  manager->request_replan(2, 1, 0., x.view(), policy);
  check(entered.wait_for(2s) == std::future_status::ready, "Solve starts");
  auto shutdown = std::async(std::launch::async, [&] { manager.reset(); });
  const bool waiting = shutdown.wait_for(50ms) == std::future_status::timeout;
  const bool plant_alive = !destroyed->load();
  optimizer->gate->release.set_value();
  check(shutdown.wait_for(2s) == std::future_status::ready, "Shutdown joins finished solve");
  shutdown.get();
  check(waiting && plant_alive && destroyed->load(),
        "Busy teardown waits before releasing worker resources");
  std::cout << "Normal replanning and in-flight shutdown passed\n";
}
