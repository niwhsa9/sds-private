#pragma once
#include <Eigen/Dense>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "sds_core.cuh"
#include "sds_tvlqr.cuh"
#include "tensor.cuh"

namespace sds
{

// This class is responsible for safely managing a thread that computes new
// policies in the background. The user can request replans which are dispatched
// to the background thread, and query for new policies as they become
// available.
template <RolloutProvider<float> P, typename PolicyOptimizer>
class ReplanManager
{
 public:
  // TODO(amg) maybe just move the PolicyOptimizer into this like the plant
  ReplanManager(P&& plant, std::shared_ptr<PolicyOptimizer> p)
      : plant(std::move(plant)), optimizer(p)
  {
    replan_thread = std::thread(
        [this]()
        {
          while (true)
          {
            std::unique_lock<std::mutex> lock(replan_mutex);
            replan_cv.wait(lock, [&]() { return stopping || state == REQUESTED; });
            if (stopping) return;
            lock.unlock();
            auto plan_start_time = std::chrono::steady_clock::now();
            auto policy = replan();
            lock.lock();
            if (stopping) return;
            last_replan_time =
                std::chrono::steady_clock::now() - plan_start_time;
            new_policy = std::move(policy);
            state = READY;
            replan_cv
                .notify_all();  // notify any threads that were blocked waiting
                                // for a new policy to become available, only
                                // matters for request_replan_blocking
          }
        });
  }

  ~ReplanManager()
  {
    {
      std::lock_guard<std::mutex> lock(replan_mutex);
      stopping = true;
    }
    replan_cv.notify_all();
    // Finish an in-flight solve before destroying its plant/optimizer or CV.
    if (replan_thread.joinable()) replan_thread.join();
  }

  // Triggers a replan and invalidates the internally-held policy if one exists
  // Note, it is an error to call this if the state is not READY, and
  // it is up to the caller to verify that with `ready_to_replan()`
  void request_replan(
      size_t horizon, size_t project_steps, double cur_time,
      const TensorView<float, 1>& x0, const LinearPolicy& current_policy)
  {
    replan_mutex.lock();
    if (state != READY)
      throw std::runtime_error("Called request_replan when not READY");
    replan_mutex.unlock();

    Tensor<float, 1> x0_copy(x0.shape(0));
    x0_copy.deep_copy_from(x0);  // meh
    current_request = std::make_unique<ReplanRequest>(
        horizon, project_steps, cur_time, std::move(x0_copy), current_policy);
    state = REQUESTED;
    replan_cv.notify_all();
  }

  std::unique_ptr<LinearPolicy> replan()
  {
    // Project state forward with the current policy to account for the time it
    // actually takes to compute a new policy
    auto [x_proj, u_proj] = simulate_plant_with_policy(
        plant, current_request->current_policy, current_request->x0.view(),
        current_request->current_policy.dt, current_request->request_time,
        current_request->project_steps);

    // Optimize a new policy from the projected state
    double projected_start_time =
        current_request->request_time +
        current_request->project_steps * current_request->current_policy.dt;
    return std::make_unique<LinearPolicy>(optimizer->operator()(
        x_proj.template slice_1d<1>(current_request->project_steps),
        projected_start_time, &current_request->current_policy));
  }

  // Checks if the manager is in the READY state
  bool ready_to_replan()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    return state == READY;
  }

  // Returns the start time for which a new policy is available, if no policy is
  // avialable, returns std::nullopt
  std::optional<double> get_available_plan_start_time()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    if (new_policy)
      return std::make_optional(new_policy->start_time);
    else
      return std::nullopt;
  }

  // This moves from the internal policy, invalidating it
  std::unique_ptr<LinearPolicy> get_policy()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    if (state != READY)
      throw std::runtime_error("Called get_policy when not READY");
    if (!new_policy)
      throw std::runtime_error(
          "Called get_policy when no new policy is available");
    return std::move(new_policy);
  }

  std::unique_ptr<LinearPolicy> get_policy_blocking()
  {
    std::unique_lock<std::mutex> lock(replan_mutex);
    replan_cv.wait(lock, [&]() { return state == READY && new_policy; });
    return std::move(new_policy);
  }

  double get_last_replan_time()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    return last_replan_time.count();
  }

 private:
  enum State
  {
    // Planner can accept a request_replan
    READY,
    // Planner is computing a new policy, and will not accept a request_replan
    REQUESTED
  };

  struct ReplanRequest
  {
    size_t horizon;
    size_t project_steps;
    double request_time;
    Tensor<float, 1> x0;
    LinearPolicy current_policy;
  };

  std::unique_ptr<LinearPolicy> new_policy;
  std::mutex replan_mutex;
  std::condition_variable replan_cv;
  State state = READY;

  // internal state for the replan request
  std::unique_ptr<ReplanRequest> current_request;
  std::shared_ptr<PolicyOptimizer> optimizer;
  P plant;

  std::chrono::duration<double> last_replan_time;
  bool stopping = false;
  std::thread replan_thread;
};

// This is just a helper function for the most common replanning loop logic. The
// loop will request replans as quickly as possible and update the user's policy
// pointer as new plans become available. Regardless, the loop returns the
// command from the most recent available policy, which may either be the
// currently supplied policy or the newly computed policy
// Usage: while(true) { u = replan_iterate(...); send_to_motors(u);}
template <typename P, typename PolicyOptimizer>
Tensor<float, 1> replan_iterate(
    size_t horizon, size_t project_steps, double cur_time,
    const TensorView<float, 1>& x_cur,
    std::unique_ptr<LinearPolicy>& cur_policy,
    ReplanManager<P, PolicyOptimizer>& replan_manager)

{
  // Check if a new plan is available and swap over if it is and the start time
  // is in the past
  std::optional<double> maybe_available_plan_start_time =
      replan_manager.get_available_plan_start_time();

  if (maybe_available_plan_start_time &&
      *maybe_available_plan_start_time <= cur_time)
  {
    // swap over to the new policy
    cur_policy = std::move(replan_manager.get_policy());

    if (*maybe_available_plan_start_time - cur_time < -0.02)
    {
      std::cout
          << "Warning: new plan is available but start time is in the past by "
          << cur_time - *maybe_available_plan_start_time << " seconds"
          << std::endl;
    }
  }

  // Request a new replan if we can and theres no new plan
  if (replan_manager.ready_to_replan() && !maybe_available_plan_start_time)
  {
    replan_manager.request_replan(
        horizon, project_steps, cur_time, x_cur, *cur_policy);
  }
  // Evaluate the current policy at the current time
  return cur_policy->eval_policy(cur_time, x_cur.data());
}

}  // namespace sds