#include "sds_tvlqr.cuh"

Tensor<float, 2> sds::keep_tape_after_time(
    const TensorView<float, 2>& x_seq, double start_time, double t, double dt,
    int pad_to_T)
{
  int step = static_cast<int>(std::floor((t - start_time) / dt));
  if (step < 0 || step >= x_seq.shape(0))
    throw std::runtime_error(
        "Time t is out of bounds for policy evaluation. t = " +
        std::to_string(t) + ", start_time = " + std::to_string(start_time) +
        ", dt = " + std::to_string(dt) + ", x_seq.shape(0) = " +
        std::to_string(x_seq.shape(0)) + ", step = " + std::to_string(step));

  if (pad_to_T == -1) pad_to_T = x_seq.shape(0) - step;
  if (pad_to_T <= 0)
    throw std::runtime_error(
        "pad_to_T must be positive. pad_to_T = " + std::to_string(pad_to_T));
  if (pad_to_T < x_seq.shape(0) - step)
  {
    std::cerr << "Warning: pad_to_T is less than the number of steps to keep. "
              << pad_to_T << std::endl;
  }

  Tensor<float, 2> result(Memory::Host, pad_to_T, x_seq.shape(1));
  result.fill(0.0f);
  for (int i = 0; i < x_seq.shape(1); ++i)
    for (int j = 0; j < std::min(x_seq.shape(0) - step, pad_to_T); ++j)
      result(j, i) = x_seq(j + step, i);

  return result;
}

std::pair<Tensor<float, 3>, Tensor<float, 3>> sds::compute_tvlqr_gains(
    const TensorView<float, 3>& A, const TensorView<float, 3>& B,
    const TensorView<float, 2>& Q, const TensorView<float, 2>& Qf,
    const TensorView<float, 2>& R, float dt, bool discretize)
{
  const int N = A.shape(0);
  const int nX = A.shape(1);
  const int nU = B.shape(2);

  if (A.shape(2) != nX) throw std::runtime_error("A must be N x nX x nX");
  if (B.shape(0) != N || B.shape(1) != nX)
    throw std::runtime_error("B must be N x nX x nU");
  if (Q.shape(0) != nX || Q.shape(1) != nX)
    throw std::runtime_error("Q must be nX x nX");
  if (Qf.shape(0) != nX || Qf.shape(1) != nX)
    throw std::runtime_error("Qf must be nX x nX");
  if (R.shape(0) != nU || R.shape(1) != nU)
    throw std::runtime_error("R must be nU x nU");

  using MatX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMajMatX =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using RowMajRef = Eigen::Map<const RowMajMatX>;

  // Map the float tensor data, then convert to double.
  using FloatRowMajMatX =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using FloatRowMajRef = Eigen::Map<const FloatRowMajMatX>;

  FloatRowMajRef Q_float(Q.data(), nX, nX);
  FloatRowMajRef Qf_float(Qf.data(), nX, nX);
  FloatRowMajRef R_float(R.data(), nU, nU);

  MatX Q_eig = Q_float.cast<double>();
  MatX Qf_eig = Qf_float.cast<double>();
  MatX R_eig = R_float.cast<double>();

  const double dt_d = static_cast<double>(dt);

  // Output remains float.
  Tensor<float, 3> K(Memory::Host, N, nX, nU);
  Tensor<float, 3> Ps(Memory::Host, N + 1, nX, nX);
  Eigen::Map<FloatRowMajMatX>(Ps.slice<0>(N).data(), nX, nX) = Qf_float;

  MatX P = Qf_eig;

  for (int i = N - 1; i >= 0; --i)
  {
    FloatRowMajRef A_i_float(A.slice<0>(i).data(), nX, nX);
    FloatRowMajRef B_i_float(B.slice<0>(i).data(), nX, nU);

    MatX A_i = A_i_float.cast<double>();
    MatX B_i = B_i_float.cast<double>();

    MatX Ak, Bk;

    if (discretize)
    {
      Ak = A_i * dt_d + MatX::Identity(nX, nX);
      Bk = B_i * dt_d;
    }
    else
    {
      Ak = A_i;
      Bk = B_i;
    }

    MatX BtPB = Bk.transpose() * P * Bk + R_eig * dt_d;
    MatX BtPA = Bk.transpose() * P * Ak;
    MatX K_i = -BtPB.inverse() * BtPA;

    // K_i is double, so cast back to float when writing to K.
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        K.slice<0>(i).data(), nX, nU) = K_i.transpose().cast<float>();

    P = Ak.transpose() * (P - P * Bk * BtPB.inverse() * Bk.transpose() * P) *
            Ak +
        Q_eig * dt_d;
    Eigen::Map<FloatRowMajMatX>(Ps.slice<0>(i).data(), nX, nX) =
        P.cast<float>();
  }

  return {std::move(K), std::move(Ps)};
}

// Tensor<float, 3> sds::compute_tvlqr_gains(
//     const TensorView<float, 3>& A, const TensorView<float, 3>& B,
//     const TensorView<float, 2>& Q, const TensorView<float, 2>& Qf,
//     const TensorView<float, 2>& R, float dt, bool discretize)
//{
//   const int N = A.shape(0);
//   const int nX = A.shape(1);
//   const int nU = B.shape(2);
//
//   if (A.shape(2) != nX) throw std::runtime_error("A must be N x nX x nX");
//   if (B.shape(0) != N || B.shape(1) != nX)
//     throw std::runtime_error("B must be N x nX x nU");
//   if (Q.shape(0) != nX || Q.shape(1) != nX)
//     throw std::runtime_error("Q must be nX x nX");
//   if (Qf.shape(0) != nX || Qf.shape(1) != nX)
//     throw std::runtime_error("Qf must be nX x nX");
//   if (R.shape(0) != nU || R.shape(1) != nU)
//     throw std::runtime_error("R must be nU x nU");
//
//   using MatX = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
//   using RowMajMatX =
//       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   using RowMajRef = Eigen::Map<const RowMajMatX>;
//
//   RowMajRef Q_eig(Q.data(), nX, nX);
//   RowMajRef Qf_eig(Qf.data(), nX, nX);
//   RowMajRef R_eig(R.data(), nU, nU);
//
//   Tensor<float, 3> K(Memory::Host, N, nX, nU);
//
//   MatX P = Qf_eig;  // convert to col-major working type
//
//   for (int i = N - 1; i >= 0; --i)
//   {
//     RowMajRef A_i(A.slice<0>(i).data(), nX, nX);
//     RowMajRef B_i(B.slice<0>(i).data(), nX, nU);
//
//     MatX Ak, Bk;
//     if (discretize)
//     {
//       Ak = A_i * dt + MatX::Identity(nX, nX);
//       Bk = B_i * dt;
//     }
//     else
//     {
//       Ak = A_i;
//       Bk = B_i;
//     }
//
//     MatX BtPB = Bk.transpose() * P * Bk + R_eig * dt;
//     MatX BtPA = Bk.transpose() * P * Ak;
//     MatX K_i = -BtPB.inverse() * BtPA;  // nU x nX
//
//     // Write K_i (nU x nX, Eigen default col-major) directly into the
//     // (nX x nU) row-major slice -- reproduces the transpose trick used
//     // in LinearPolicy::eval_policy.
//     Eigen::Map<MatX>(K.slice<0>(i).data(), nU, nX) = K_i;
//
//     P = Ak.transpose() * (P - P * Bk * BtPB.inverse() * Bk.transpose() * P) *
//             Ak +
//         Q_eig * dt;
//   }
//
//   return K;
// }

Tensor<float, 2> sds::decimate_trajectory(
    const TensorView<float, 2>& x_seq, float old_dt, float new_dt)
{
  if (new_dt <= old_dt)
    throw std::runtime_error(
        "New dt must be larger than old dt for decimating.");

  int ratio = static_cast<int>(std::round(new_dt / old_dt));

  int T = x_seq.shape(0) / ratio;
  Tensor<float, 2> x_decimated(Memory::Host, T, x_seq.shape(1));

  for (int i = 0; i < T; ++i)
  {
    int idx = i * ratio;
    if (idx >= x_seq.shape(0))
      throw std::runtime_error(
          "Index out of bounds when decimating trajectory: " +
          std::to_string(idx) + " >= " + std::to_string(x_seq.shape(0)));
    x_decimated.slice_1d<1>(i).deep_copy_from(x_seq.slice_1d<1>(idx));
  }

  return x_decimated;
}
