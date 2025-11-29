#include "helper_functions.h"
#include "spmm.h"

using namespace std;

#define ROWS_PER_BLOCK NUM_PES              // process NUM_PES CPU-rows at a time

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");


void cpu_spmm(const CSRMatrix& A, const vector<vector<float>> B, vector<vector<float>> Cin, const float alpha, const float beta, vector<vector<float>>& Cout, const int M1, const int N1) {  // Initialize result vector C with zeros

// Perform matrix-vector multiplication
  for (int i = 0; i < A.row_offsets.size() - 1; ++i) {
    for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
      int colIndex = A.col_indices[j];
      float value = A.values[j];
      
      for(int k = 0; k < N1; k++) {
        Cout[i][k] += value * B[colIndex][k];
      }
    }
  }

  for (int i = 0; i < M1; i++) {
    for (int j = 0; j < N1; j++) {
      Cout[i][j] = alpha*Cout[i][j] + beta*Cin[i][j];
    }
  }

}

double computePrecisionLoss(const vector<vector<float>>& cpu, const vector<aligned_vector<float>>& fpga, const int M1, const int N1) 
{
  double diffSum = 0.0;
  double refSum = 0.0;
  double maxRelativeError = 0.0;
  int maxRelativeErrorIdx = 0;
  float max_cpu = 0;
  float max_fpga = 0;

  // Key derived parameters for file 65's FPGA pattern
  const int CHUNKS_PER_CH_PER_BLOCK = (ROWS_PER_BLOCK / 2) / NUM_C_CH;
  const int row_blocks_per_tile = M0 / ROWS_PER_BLOCK;

  int total_row_blocks = (M1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  int numTilesM = ((M1 - 1) / M0) + 1;
  int numTilesN = N1 / N0;

  int comparison_count = 0;
  int mismatch_count = 0;
  const int MAX_MISMATCH_PRINT = 10;

  // Iterate through all blocks
  for (int block = 0; block < total_row_blocks; ++block) {
    int row_tile_idx = block / row_blocks_per_tile;
    int block_in_tile = block % row_blocks_per_tile;

    // Calculate actual blocks in this row tile (last tile may be partial)
    int blocks_in_this_tile;
    if (row_tile_idx < numTilesM - 1) {
      blocks_in_this_tile = row_blocks_per_tile;
    } else {
      blocks_in_this_tile = total_row_blocks - row_tile_idx * row_blocks_per_tile;
    }

    // Base offset: all previous row tiles
    int chunks_base = row_tile_idx * numTilesN * row_blocks_per_tile * CHUNKS_PER_CH_PER_BLOCK;

    // Iterate over all column tiles
    for (int col_tile_idx = 0; col_tile_idx < numTilesN; ++col_tile_idx) {
      int chunks_per_col_in_this_tile = blocks_in_this_tile * CHUNKS_PER_CH_PER_BLOCK;
      int col_tile_offset = col_tile_idx * chunks_per_col_in_this_tile;

      for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        int global_row = block * ROWS_PER_BLOCK + r;
        if (global_row >= M1) continue;

        for (int c = 0; c < N0; ++c) {
          int global_col = col_tile_idx * N0 + c;
          if (global_col >= N1) continue;

          // Channel assignment: consecutive rows per channel
          // Each channel handles ROWS_PER_BLOCK/NUM_C_CH consecutive rows
          const int ROWS_PER_CHANNEL = ROWS_PER_BLOCK / NUM_C_CH;
          int ch = r / ROWS_PER_CHANNEL;  // channel based on row range
          int r_in_channel = r % ROWS_PER_CHANNEL;  // row index within channel
          
          // Chunk within channel (2 rows per chunk)
          int chunk_in_channel = r_in_channel / 2;
          
          // Row position within the chunk (0=even row, 1=odd row)
          int row_in_chunk = r_in_channel % 2;
          
          // Chunk index within this channel for this block
          int chunk_idx_in_ch = chunk_in_channel;

          // Block offset within the column tile
          int block_offset = block_in_tile * CHUNKS_PER_CH_PER_BLOCK;

          // Final chunk index in this channel's memory
          int local_chunk_idx = chunks_base + col_tile_offset + block_offset + chunk_idx_in_ch;
          
          // Offset within the 16-float chunk
          int offset_in_chunk = row_in_chunk * N0 + c;
          int addr = local_chunk_idx * B_CHUNK_SIZE + offset_in_chunk;

          // Get values and compute error
          float fpga_val = fpga[ch][addr];
          float cpu_val = cpu[global_row][global_col];

          double diff = fabs(fpga_val - cpu_val);
          double relativeError = 0.0;

          if (cpu_val != 0) {
            relativeError = diff / fabs(cpu_val);
            if (relativeError > maxRelativeError) {
              maxRelativeErrorIdx = comparison_count;
              maxRelativeError = relativeError;
              max_cpu = cpu_val;
              max_fpga = fpga_val;
            }
          }


          diffSum += diff;
          refSum += fabs(cpu_val);
          comparison_count++;
        }
      }
    }
  }


  return diffSum/refSum;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  if (argc < 2){
    clog << "Invalid Arguments" << endl;
    return 0; 
  }

  char* filename = argv[1];
  uint16_t rp_time = 1;
  uint16_t N = 128;

  if (argc == 3) 
    rp_time = (uint16_t)atoi(argv[2]);

  else if (argc == 4) {
    rp_time = (uint16_t)atoi(argv[2]);
    N = (uint16_t)atoi(argv[3]);
  }


  const float alpha = 1.00;
  const float beta = -2.00;

  clog << "Reading A " << filename << endl;

  vector<float> cscValues;
  vector<int> cscRowIndices;
  vector<int> cscColOffsets;
  int M, K, nnz;
  readMatrixCSC(filename, cscValues, cscRowIndices, cscColOffsets, M, K, nnz);


  CSRMatrix cpuAmtx;
  convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets, cpuAmtx.values, cpuAmtx.col_indices, cpuAmtx.row_offsets, M, K, nnz);

  int N1 = max(N0, 1 << (int(log2(N-1))+1)); // make sure N is a power of 2 and >= N0
  int numTilesN = N1/N0; //N is always divisible by N0

  //M and K should be multiple of MX and KX
  int MX1 = std::lcm(NUM_PES, MX);
  
  
  //Padded M and K are M1 and K1
  int M1 = (((M-1)/MX1) + 1) * MX1;
  int K1 = (((K-1)/KX) + 1) * KX;

  int numTilesM = ((M1 - 1) / M0) + 1;
  int numTilesK = ((K1 - 1) / K0) + 1;

  uint32_t numRowsPerPE = M1 / NUM_PES;
 
  cout << "M: " << M << "\t N: " << N << "\t K: " << K << "\t NNZ: " << nnz << endl;
  cout << "M0: " << M0 << "\t N0: " << N0 << "\t K0: " << K0 << endl;
  cout << "M1: " << M1 << "\t N1: " << N1 << "\t K1: " << K1 << endl;
  cout << "NumtilesM: " << numTilesM << ", NumtilesK: " << numTilesK << ", NumTilesN: " << numTilesN << endl; 
  cout << "NumRowsPerPE: " << numRowsPerPE << endl << endl; 

  vector<vector<float>>cpuBinMtx(K1, vector<float>(N1, 0));
  vector<vector<float>>cpuCinMtx(M1, vector<float>(N1, 0));
  vector<vector<float>>cpuCoutMtx(M1, vector<float>(N1, 0));

  vector<aligned_vector<float>>fpgaBinMtx(NUM_B_CH, aligned_vector<float>((K1 * N1) / NUM_B_CH, 0));
  vector<aligned_vector<float>>fpgaCinMtx(NUM_C_CH, aligned_vector<float>((M1 * N1) / NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutMtx(NUM_C_CH, aligned_vector<float>((M1 * N1) / NUM_C_CH, 0));

  clog << "Generating Dense Matrices B and C..."  << endl;
  for (int i = 0; i < M1; i++) {
    for (int j = 0; j < N1; j++) {
      cpuCinMtx[i][j] = 2.0 * (i*N1+j);
    }
  }

  for (int i = 0; i < K1; i++) {
    for (int j = 0; j < N1; j++) {
      cpuBinMtx[i][j] = 1.0 *  (i*N1+j);
    }
  }

  cout << "Preparing Matrices for FPGA " << endl; 

  auto start_gen = std::chrono::steady_clock::now();
  vector<vector<CSRMatrix>> tiledMatrices = tileCSRMatrix(cpuAmtx, M, K, M0, K0, numTilesM, numTilesK);
  vector<aligned_vector<uint64_t>> fpgaAmtx = prepareAmtx(tiledMatrices, numTilesM, numTilesK, M0, K0, M, K, nnz);

  // Calculate total row blocks needed
  int total_row_blocks = (M1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  int row_blocks_per_tile = M0 / ROWS_PER_BLOCK;

 
  const int CHUNKS_PER_CH_PER_BLOCK = (ROWS_PER_BLOCK / 2) / NUM_C_CH;

  cout << "M1: " << M1 << ", N1: " << N1 << ", NUM_C_CH: " << NUM_C_CH << endl;
  cout << "M0: " << M0 << ", row_blocks_per_tile: " << row_blocks_per_tile << endl;
  cout << "numTilesM: " << numTilesM << ", numTilesN: " << numTilesN << endl;
  cout << "fpgaCinMtx size: " << fpgaCinMtx[0].size() << " floats per channel" << endl;
  cout << "fpgaAmtx size: " << fpgaAmtx[0].size() << " uint64_t per channel" << endl;
  

  // Input mapping with bit-interleaved row pattern (matching FPGA output)
  for (int block = 0; block < total_row_blocks; ++block) {
    int row_tile_idx = block / row_blocks_per_tile;
    int block_in_tile = block % row_blocks_per_tile;

    int blocks_in_this_tile;
    if (row_tile_idx < numTilesM - 1) {
      blocks_in_this_tile = row_blocks_per_tile;
    } else {
      blocks_in_this_tile = total_row_blocks - row_tile_idx * row_blocks_per_tile;
    }

    int chunks_base = row_tile_idx * numTilesN * row_blocks_per_tile * CHUNKS_PER_CH_PER_BLOCK;

    for (int col_tile_idx = 0; col_tile_idx < numTilesN; ++col_tile_idx) {
      int chunks_per_col_in_this_tile = blocks_in_this_tile * CHUNKS_PER_CH_PER_BLOCK;
      int col_tile_offset = col_tile_idx * chunks_per_col_in_this_tile;

      for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        int global_row = block * ROWS_PER_BLOCK + r;
        if (global_row >= M1) continue;

        for (int c = 0; c < N0; ++c) {
          int global_col = col_tile_idx * N0 + c;
          float val = cpuCinMtx[global_row][global_col];

          // Channel assignment: consecutive rows per channel
          const int ROWS_PER_CHANNEL = ROWS_PER_BLOCK / NUM_C_CH;
          int ch = r / ROWS_PER_CHANNEL;
          int r_in_channel = r % ROWS_PER_CHANNEL;
          
          // Chunk within channel (2 rows per chunk)
          int chunk_in_channel = r_in_channel / 2;
          
          // Row position within the chunk
          int row_in_chunk = r_in_channel % 2;
          
          // Chunk index within this channel for this block
          int chunk_idx_in_ch = chunk_in_channel;

          // Block offset
          int block_offset = block_in_tile * CHUNKS_PER_CH_PER_BLOCK;

          // Final address
          int local_chunk_idx = chunks_base + col_tile_offset + block_offset + chunk_idx_in_ch;
          int offset_in_chunk = row_in_chunk * N0 + c;
          int addr = local_chunk_idx * B_CHUNK_SIZE + offset_in_chunk;

          // Safety check
          if (ch >= NUM_C_CH || addr >= fpgaCinMtx[ch].size()) {
            cout << "Out of bounds! ch=" << ch << ", addr=" << addr 
                 << ", max=" << fpgaCinMtx[ch].size() << endl;
            exit(1);
          }

          fpgaCinMtx[ch][addr] = val;
        }
      }
    }
  }

  int linear_b_addr = 0;
  for(int i = 0; i < K1; i+=K0) {
    for(int j = 0; j < N1; j+=N0) {
      for(int ii = 0; (ii < K0) && (i+ii < K1); ii++) {
        for(int jj = 0; jj < N0; jj++) {
          int ch = (linear_b_addr / B_CHUNK_SIZE) % NUM_B_CH;
          int addr = (linear_b_addr / (NUM_B_CH * B_CHUNK_SIZE)) * B_CHUNK_SIZE + (linear_b_addr % B_CHUNK_SIZE);
          fpgaBinMtx[ch][addr] = cpuBinMtx[i+ii][j+jj];
          linear_b_addr++;
        }
      }
    }
  }

  auto end_gen = std::chrono::steady_clock::now();
  double time_gen = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
  time_gen *= 1e-9;
  cout << "Pre-processing Time: " << time_gen*1000 << " msec\n" << endl;

  cout <<  endl << "Computing CPU SpMM... "  << endl;

  auto start_cpu = std::chrono::steady_clock::now();
  cpu_spmm(cpuAmtx, cpuBinMtx, cpuCinMtx, alpha, beta, cpuCoutMtx, M1, N1);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
  time_cpu *= 1e-9;
  cout << "done (" << time_cpu*1000 << " msec)\n";
  cout << "CPU GFLOPS: " << 2.0 * (nnz + M) * N1 / 1e+9 / time_cpu << "\n" << endl;



  uint32_t run_len = fpgaAmtx[0].size()/ PES_PER_CH;
  const uint32_t last_tile_idx = (numTilesM * numTilesN * rp_time)  - 1; 

  cout <<  endl << "Computing FPGA SpMM Emulation... "  << endl;

  double time_taken = tapa::invoke(hispmm, FLAGS_bitstream,
          tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAmtx).reinterpret<uint64_v>(),
          tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinMtx).reinterpret<float_vB>(),
          tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinMtx).reinterpret<float_vB>(),
          tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutMtx).reinterpret<float_vB>(),
          alpha, beta,
          (uint32_t)M1, (uint32_t)N1, (uint32_t)K1,
          (uint32_t)numTilesM, (uint32_t)numTilesN, (uint32_t)numTilesK,
          run_len, last_tile_idx, rp_time);

 
  time_taken *= (1e-9); // total time in second
  time_taken /= rp_time;
    printf("Kernel time:%f\n", time_taken*1000);
  float gflops =
    2.0 * (nnz + M)*N1
    / 1e+9
    / time_taken
    ;
  printf("GFLOPS:%f\n", gflops);



  cout <<  endl << "Comparing Results... "  << endl;
  double precisionLoss = computePrecisionLoss(cpuCoutMtx, fpgaCoutMtx, M1, N1);
  // cout << "Precision Loss: " << precisionLoss << endl;
  if (precisionLoss < 1e-4) {
    cout << "Success!" << endl;
  } else {
    cout << "Failure! Precision Loss: " << precisionLoss << endl;
  }
  return 0;

}

