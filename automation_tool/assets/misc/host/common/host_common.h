// A4-new host common helpers (host-only).
//
// This header consolidates:
// - cpu_ref.h
// - host_args.h
// - problem_setup.h
// - init_dense.h
// - pack_b.h
// - report_precision_loss.h
//
// Keep this out of kernel-facing headers.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Defined in `hispmm_host.cpp` (host main). Headers should DECLARE before use.
DECLARE_bool(verbose);

#include "prepare_amt_unified.h"
#include "hispmm.h"
#include "prepare_fpga_cin_unified.h"
#include "compare_fpga_c_unified.h"

// -----------------------------
// CPU reference (SpMM on CPU)
// -----------------------------
inline void cpu_spmm(const CSRMatrix& A,
                     const std::vector<std::vector<float>> B,
                     const std::vector<std::vector<float>> Cin,
                     const float alpha,
                     const float beta,
                     std::vector<std::vector<float>>& Cout,
                     const int M1,
                     const int N1) {
  for (int i = 0; i < static_cast<int>(A.row_offsets.size()) - 1; ++i) {
    for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
      const int colIndex = A.col_indices[j];
      const float value = A.values[j];
      for (int k = 0; k < N1; k++) {
        Cout[i][k] += value * B[colIndex][k];
      }
    }
  }
  for (int i = 0; i < M1; i++) {
    for (int j = 0; j < N1; j++) {
      Cout[i][j] = alpha * Cout[i][j] + beta * Cin[i][j];
    }
  }
}

// -----------------------------
// CLI args
// -----------------------------
struct HostArgs {
  std::string matrix_path;
  uint16_t rp_time = 1;
  uint16_t N = 128;
  float alpha = 1.0f;
  float beta = 2.0f;
};

inline void PrintUsage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << (prog ? prog : "hispmm") << " <matrix.mtx> [rp_time] [N] [alpha] [beta]\n\n"
            << "Defaults:\n"
            << "  rp_time = 1\n"
            << "  N       = 128\n"
            << "  alpha   = 1.0\n"
            << "  beta    = 2.0\n";
}

inline bool ParseHostArgs(int argc, char* argv[], HostArgs* out) {
  if (out == nullptr) return false;
  if (argc < 2 || argc > 6) {
    PrintUsage(argc > 0 ? argv[0] : "hispmm");
    return false;
  }
  out->matrix_path = argv[1];
  if (argc >= 3) out->rp_time = static_cast<uint16_t>(std::atoi(argv[2]));
  if (argc >= 4) out->N = static_cast<uint16_t>(std::atoi(argv[3]));
  if (argc >= 5) out->alpha = std::strtof(argv[4], nullptr);
  if (argc >= 6) out->beta = std::strtof(argv[5], nullptr);
  return true;
}

// -----------------------------
// Problem setup (load A + derive dims)
// -----------------------------
struct ProblemSetup {
  int M = 0;
  int K = 0;
  int nnz = 0;

  int N_in = 0;
  int N1 = 0;
  int M1 = 0;
  int K1 = 0;
  int numTilesN = 0;
  int numTilesM = 0;
  int numTilesK = 0;
  uint32_t numRowsPerPE = 0;

  CSRMatrix cpuAmtx;
};

inline ProblemSetup LoadAAndDeriveProblem(const char* filename, uint16_t N_requested) {
  ProblemSetup out;
  out.N_in = static_cast<int>(N_requested);

  if (FLAGS_verbose) {
    std::cout << "Reading A " << filename << std::endl;
  }

  std::vector<float> cscValues;
  std::vector<int> cscRowIndices;
  std::vector<int> cscColOffsets;
  readMatrixCSC(const_cast<char*>(filename), cscValues, cscRowIndices, cscColOffsets, out.M, out.K, out.nnz);

  convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets,
                  out.cpuAmtx.values, out.cpuAmtx.col_indices, out.cpuAmtx.row_offsets,
                  out.M, out.K, out.nnz);

  // Pad N to next power-of-two and ensure >= N0.
  int n_pow2 = 1;
  while (n_pow2 < std::max(1, out.N_in)) n_pow2 <<= 1;
  out.N1 = std::max(N0, n_pow2);
  out.numTilesN = out.N1 / N0;

  const int MX1 = std::lcm(NUM_PES, MX);
  out.M1 = (((out.M - 1) / MX1) + 1) * MX1;
  out.K1 = (((out.K - 1) / KX) + 1) * KX;

  out.numTilesM = ((out.M1 - 1) / M0) + 1;
  out.numTilesK = ((out.K1 - 1) / K0) + 1;
  out.numRowsPerPE = static_cast<uint32_t>(out.M1 / NUM_PES);

  if (FLAGS_verbose) {
    std::cout << "M: " << out.M << "\t N: " << out.N_in << "\t K: " << out.K << "\t NNZ: " << out.nnz << "\n";
    std::cout << "M0: " << M0 << "\t N0: " << N0 << "\t K0: " << K0 << "\n";
    std::cout << "M1: " << out.M1 << "\t N1: " << out.N1 << "\t K1: " << out.K1 << "\n";
    std::cout << "NumtilesM: " << out.numTilesM << ", NumtilesK: " << out.numTilesK << ", NumTilesN: " << out.numTilesN
              << "\n";
    std::cout << "NumRowsPerPE: " << out.numRowsPerPE << "\n\n";
  }

  return out;
}

// -----------------------------
// Dense init (B and Cin)
// -----------------------------
inline void InitDenseBC(std::vector<std::vector<float>>& cpuBinMtx,
                        std::vector<std::vector<float>>& cpuCinMtx,
                        int K1,
                        int M1,
                        int N1) {
  for (int i = 0; i < M1; i++) {
    for (int j = 0; j < N1; j++) {
      cpuCinMtx[i][j] = 2.0f * (i * N1 + j);
    }
  }
  for (int i = 0; i < K1; i++) {
    for (int j = 0; j < N1; j++) {
      cpuBinMtx[i][j] = 1.0f * (i * N1 + j);
    }
  }
}

// -----------------------------
// Pack B (linear interleave)
// -----------------------------
inline void PackBLinearInterleave(const std::vector<std::vector<float>>& cpuBinMtx,
                                  std::vector<aligned_vector<float>>& fpgaBinMtx,
                                  int K1,
                                  int N1) {
  const int b_chunk_size = B_CHUNK_SIZE;  // 2*N0
  int linear_b_addr = 0;
  for (int i = 0; i < K1; i += K0) {
    for (int j = 0; j < N1; j += N0) {
      for (int ii = 0; (ii < K0) && (i + ii < K1); ii++) {
        for (int jj = 0; jj < N0; jj++) {
          const int ch = (linear_b_addr / b_chunk_size) % NUM_B_CH;
          const int addr = (linear_b_addr / (NUM_B_CH * b_chunk_size)) * b_chunk_size +
                           (linear_b_addr % b_chunk_size);
          fpgaBinMtx[ch][addr] = cpuBinMtx[i + ii][j + jj];
          linear_b_addr++;
        }
      }
    }
  }
}

// -----------------------------
// Precision-loss reporting
// -----------------------------
inline void ReportPrecisionLossBalanced(double precision_loss) {
  if (precision_loss == 0.0) {
    std::cout << "Success!" << std::endl;
  } else {
    std::cout << "Failure! Precision Loss: " << precision_loss << std::endl;
  }
}

inline void ReportPrecisionLossImbalanced(double precision_loss) {
  constexpr double kThreshold = 1e-4;
  if (precision_loss < kThreshold) {
    std::cout << "Success!!" << std::endl;
  } else {
    std::cout << "Failure!! Precision Loss: " << precision_loss << std::endl;
  }
}

// Namespaced wrappers (so call sites can use `hispmm_host::...` consistently).
namespace hispmm_host {
inline void ReportPrecisionLossBalanced(double precision_loss) { ::ReportPrecisionLossBalanced(precision_loss); }
inline void ReportPrecisionLossImbalanced(double precision_loss) { ::ReportPrecisionLossImbalanced(precision_loss); }
}  // namespace hispmm_host


