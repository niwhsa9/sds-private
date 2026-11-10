#include <iostream>

#include "sds_tvlqr.cuh"

void check(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}
template <typename F>
void rejects(F operation)
{
  bool threw = false;
  try
  {
    operation();
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  check(threw, "Expected rejection");
}
int main()
{
  Tensor<float, 2> A(Memory::Host, 1, 1), B(Memory::Host, 1, 1),
      Q(Memory::Host, 1, 1), R(Memory::Host, 1, 1);
  A(0, 0) = B(0, 0) = Q(0, 0) = R(0, 0) = 1;
  auto [K, P] = sds::compute_infinite_horizon_lqr_gains(A, B, Q, R, 1, false);
  check(std::abs(P(0, 0) - 1.61803398875) < 1e-6, "Scalar DARE cost");
  check(std::abs(K(0, 0) + 0.61803398875) < 1e-6, "Scalar DARE gain/sign");
  rejects([&] { sds::compute_infinite_horizon_lqr_gains(A, B, Q, R, 0); });
  R(0, 0) = -1;
  rejects([&]
          { sds::compute_infinite_horizon_lqr_gains(A, B, Q, R, 1, false); });
  R(0, 0) = 1;
  B(0, 0) = 0;
  A(0, 0) = 1.1;
  rejects(
      [&] {
        sds::compute_infinite_horizon_lqr_gains(A, B, Q, R, 1, false, 1e-10,
                                                10);
      });
  Q(0, 0) = 0;
  rejects([&]
          { sds::compute_infinite_horizon_lqr_gains(A, B, Q, R, 1, false); });

  // Coupled pitch-rate/actuator example: compare against an independently
  // evaluated long finite-horizon recursion, including state-major layout.
  Tensor<float, 2> Ac(Memory::Host, 2, 2), Bc(Memory::Host, 2, 1),
      Qc(Memory::Host, 2, 2);
  Ac.fill(0);
  Bc.fill(0);
  Qc.fill(0);
  Ac(0, 0) = -0.2;
  Ac(0, 1) = 2;
  Ac(1, 1) = -3;
  Bc(1, 0) = 3;
  Qc(0, 0) = 10;
  Qc(1, 1) = 0.1;
  auto [Kc, Pc] = sds::compute_infinite_horizon_lqr_gains(Ac, Bc, Qc, R, 0.01);
  Eigen::Matrix2d ad, p = Eigen::Matrix2d::Zero(), q;
  ad << 0.998, 0.02, 0, 0.97;
  q << 0.1, 0, 0, 0.001;
  Eigen::Vector2d bd(0, 0.03);
  Eigen::RowVector2d gain;
  for (int i = 0; i < 10000; ++i)
  {
    gain = -(bd.transpose() * p * ad) / (0.01 + (bd.transpose() * p * bd)(0));
    p = (q + ad.transpose() * p * ad + ad.transpose() * p * bd * gain).eval();
  }
  check(std::abs(Kc(0, 0) - gain(0)) < 1e-5 &&
            std::abs(Kc(1, 0) - gain(1)) < 1e-5,
        "Coupled gain/layout");
  check(std::abs(Pc(0, 1) - p(0, 1)) < 1e-5, "Coupled Riccati cost");

  // Two inputs with unequal penalties catch transposed gain storage.
  Tensor<float, 2> Am(Memory::Host, 2, 2), Bm(Memory::Host, 2, 2),
      Qm(Memory::Host, 2, 2), Rm(Memory::Host, 2, 2);
  Am.fill(0);
  Bm.fill(0);
  Qm.fill(0);
  Rm.fill(0);
  Am(0, 0) = Am(1, 1) = Bm(0, 1) = Bm(1, 0) = 1;
  Qm(0, 0) = Qm(1, 1) = Rm(0, 0) = 1;
  Rm(1, 1) = 4;
  auto [Km, Pm] =
      sds::compute_infinite_horizon_lqr_gains(Am, Bm, Qm, Rm, 1, false);
  check(std::abs(Km(1, 0) + 0.61803398875) < 1e-6 &&
            std::abs(Km(0, 1) + 0.39038820320) < 1e-6 && Km(0, 0) == 0 &&
            Km(1, 1) == 0,
        "Multiple-input gain layout");
  Tensor<float, 2> bad(Memory::Host, 3, 2);
  rejects([&] { sds::compute_infinite_horizon_lqr_gains(Am, bad, Qm, Rm, 1); });

  Tensor<float, 1> xr(Memory::Host, 1), ur(Memory::Host, 1);
  xr(0) = ur(0) = 0;
  sds::InfiniteHorizonPolicy fallback(std::move(K), std::move(P), std::move(xr),
                                      std::move(ur));
  fallback.u_lower.emplace(Memory::Host, 1);
  (*fallback.u_lower)(0) = -0.25;
  float state[] = {1};
  check(fallback.eval_policy(state)(0) == -0.25f, "Input saturation");

  // One-state policy also proves the fallback bypasses legacy quaternion
  // handling, which is specific to the finite tape's current implementation.
  Tensor<float, 3> kt(Memory::Host, 2, 1, 1), pt(Memory::Host, 3, 1, 1);
  Tensor<float, 2> xt(Memory::Host, 3, 1), ut(Memory::Host, 2, 1);
  kt.fill(0);
  pt.fill(0);
  xt.fill(0);
  ut.fill(0);
  sds::LinearPolicy tape(std::move(kt), std::move(pt), std::move(xt),
                         std::move(ut), 1, 0.25);
  tape.terminal_policy.emplace(std::move(fallback));
  check(tape.eval_policy(1.5, state)(0) == -0.25f, "Exact tape expiry");
  check(tape.eval_policy(100, state)(0) == -0.25f,
        "Infinite fallback duration");
  int devices = 0;
  if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0)
  {
    auto copy = tape;
    check(copy.eval_policy(1.5, state)(0) == -0.25f, "Copied terminal policy");
    check(tape.eval_policy(0, state)(0) == 0, "Pre-start behavior");
    tape.terminal_policy.reset();
    check(tape.eval_policy(1.5, state)(0) == 0, "Unconfigured expiry behavior");
  }
  else
    std::cout << "Skipped legacy allocation/copy checks: no CUDA device\n";
  std::cout << "Infinite-horizon LQR checks passed\n";
}
