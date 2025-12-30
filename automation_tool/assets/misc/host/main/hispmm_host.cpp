#include "prepare_amt_unified.h"
#include "host_common.h"
#include "hispmm.h"

#include <chrono>
#include <gflags/gflags.h>

using namespace std;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_bool(verbose, false, "Enable verbose host logging (status prints + A-matrix summary).");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  // @CODEGEN:HOST_MAIN_BEGIN
  HostArgs args;
  if (!ParseHostArgs(argc, argv, &args)) {
    return 0; 
  }

  const char* filename = args.matrix_path.c_str();
  const uint16_t rp_time = args.rp_time;
  const uint16_t N = args.N;
  const float alpha = args.alpha;
  const float beta = args.beta;

  const ProblemSetup setup = LoadAAndDeriveProblem(filename, N);
  const int M = setup.M;
  const int K = setup.K;
  const int nnz = setup.nnz;
  const int N1 = setup.N1;
  const int numTilesN = setup.numTilesN;
  const int M1 = setup.M1;
  const int K1 = setup.K1;
  const int numTilesM = setup.numTilesM;
  const int numTilesK = setup.numTilesK;
  const uint32_t numRowsPerPE = setup.numRowsPerPE;
  CSRMatrix cpuAmtx = setup.cpuAmtx;

  vector<vector<float>>cpuBinMtx(K1, vector<float>(N1, 0));
  vector<vector<float>>cpuCinMtx(M1, vector<float>(N1, 0));
  vector<vector<float>>cpuCoutMtx(M1, vector<float>(N1, 0));

  vector<aligned_vector<float>>fpgaBinMtx(NUM_B_CH, aligned_vector<float>((K1 * N1) / NUM_B_CH, 0));
  vector<aligned_vector<float>>fpgaCinMtx(NUM_C_CH, aligned_vector<float>((M1 * N1) / NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutMtx(NUM_C_CH, aligned_vector<float>((M1 * N1) / NUM_C_CH, 0));

  if (FLAGS_verbose) cout << "Generating Dense Matrices B and C..."  << endl;
  InitDenseBC(cpuBinMtx, cpuCinMtx, /*K1=*/K1, /*M1=*/M1, /*N1=*/N1);

  if (FLAGS_verbose) cout << "Preparing Matrices for FPGA " << endl; 

  auto start_gen = std::chrono::steady_clock::now();
  vector<vector<CSRMatrix>> tiledMatrices = tileCSRMatrix(cpuAmtx, M, K, M0, K0, numTilesM, numTilesK);

  // @CODEGEN:PREPARE_A_CONFIG_BEGIN
  // Unified A-matrix packing.
  hispmm_host::PrepareAmtxConfig a_cfg;
  a_cfg.kernel_supports_row_sharing = true;
  a_cfg.row_sharing = hispmm_host::RowSharingPolicy::kAuto;
  a_cfg.shared_row_limit = -1;
  a_cfg.delta_improvement_threshold_percent = 25.0;
  a_cfg.print_summary = FLAGS_verbose;
  // @CODEGEN:PREPARE_A_CONFIG_END

  // @CODEGEN:PREPARE_A_CALL_BEGIN
  const hispmm_host::PrepareAmtxResult ares =
      hispmm_host::PrepareAmtxUnified(tiledMatrices, /*numTilesRows=*/numTilesM, /*numTilesCols=*/numTilesK,
                                      /*tile_depth=*/M0, /*nnz_total=*/nnz, a_cfg);
  vector<aligned_vector<uint64_t>> fpgaAmtx = ares.fpgaAmtx;
  // @CODEGEN:PREPARE_A_CALL_END



  // @CODEGEN:PREPARE_C_CONFIG_BEGIN
  //since N0 is small N1 is padded to be always a multiple of N0, but M1, K1 is not always a multiple of M0, K0
  hispmm_host::CinPrepareConfig c_cfg;
  c_cfg.layout = hispmm_host::CinLayout::LinearChunkInterleave;
  c_cfg.M1 = M1;
  c_cfg.N1 = N1;
  c_cfg.num_c_ch = NUM_C_CH;
  c_cfg.rows_per_block = NUM_PES;  // ok
  c_cfg.n0 = N0;
  c_cfg.b_chunk_size = 2 * N0;
  c_cfg.m0 = M0;
  // @CODEGEN:PREPARE_C_CONFIG_END

  // @CODEGEN:PREPARE_C_CALL_BEGIN
  hispmm_host::PrepareFpgaCinMtx(cpuCinMtx, fpgaCinMtx, c_cfg);
  // @CODEGEN:PREPARE_C_CALL_END

  // @CODEGEN:PREPARE_B_CALL_BEGIN
  PackBLinearInterleave(cpuBinMtx, fpgaBinMtx, /*K1=*/K1, /*N1=*/N1);
  // @CODEGEN:PREPARE_B_CALL_END

  auto end_gen = std::chrono::steady_clock::now();
  double time_gen = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
  time_gen *= 1e-9;
  if (FLAGS_verbose) cout << "Pre-processing Time: " << time_gen*1000 << " msec\n" << endl;

  if (FLAGS_verbose) cout <<  endl << "Computing CPU SpMM... "  << endl;

  auto start_cpu = std::chrono::steady_clock::now();
  cpu_spmm(cpuAmtx, cpuBinMtx, cpuCinMtx, alpha, beta, cpuCoutMtx, M1, N1);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
  time_cpu *= 1e-9;
  if (FLAGS_verbose) {
    cout << "done (" << time_cpu*1000 << " msec)\n";
    cout << "CPU GFLOPS: " << 2.0 * (nnz + M) * N1 / 1e+9 / time_cpu << "\n" << endl;
  }



  // @CODEGEN:RUN_CONTROL_BEGIN
  uint32_t run_len = static_cast<uint32_t>(ares.run_len);
  
  const uint32_t last_tile_idx = (numTilesM * numTilesN * rp_time)  - 1; 
  // @CODEGEN:RUN_CONTROL_END

  if (FLAGS_verbose) cout <<  endl << "Computing FPGA SpMM Emulation... "  << endl;

  // @CODEGEN:KERNEL_INVOKE_BEGIN
  tapa::invoke(hispmm, FLAGS_bitstream,
          tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAmtx).reinterpret<uint64_v>(),
          tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinMtx).reinterpret<float_vB>(),
          tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinMtx).reinterpret<float_vB>(),
          tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutMtx).reinterpret<float_vB>(),
          alpha, beta,
          (uint32_t)M1, (uint32_t)N1, (uint32_t)K1,
          (uint32_t)numTilesM, (uint32_t)numTilesN, (uint32_t)numTilesK,
          run_len, last_tile_idx, rp_time);
  // @CODEGEN:KERNEL_INVOKE_END


  if (FLAGS_verbose) cout <<  endl << "Comparing Results... "  << endl;



double precisionLoss = hispmm_host::ComputePrecisionLossUnified(cpuCoutMtx, fpgaCoutMtx, c_cfg);
  // @CODEGEN:REPORT_PRECISION_LOSS_BEGIN
  if (ares.used_row_sharing) {
    hispmm_host::ReportPrecisionLossImbalanced(precisionLoss);
  } else {
    hispmm_host::ReportPrecisionLossBalanced(precisionLoss);
  }
  // @CODEGEN:REPORT_PRECISION_LOSS_END
  return 0;

  // @CODEGEN:HOST_MAIN_END
}