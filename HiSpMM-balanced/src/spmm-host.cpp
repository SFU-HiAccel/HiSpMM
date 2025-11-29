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

  int total_row_blocks = (M1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  int numTilesN = N1 / N0;  // Number of column tiles
  int comparison_count = 0;
  int mismatch_count = 0;  // Debug: count mismatches
  const int MAX_MISMATCH_PRINT = 10;  // Debug: print first 10 mismatches
  
  const int ROWS_PER_CHANNEL = ROWS_PER_BLOCK / NUM_C_CH; // 20 rows per channel
  const int HALF_GROUP = ROWS_PER_CHANNEL / 2; // 10 rows per half group
  const int CHUNKS_PER_CHANNEL = ROWS_PER_CHANNEL / 2; // Exactly 10 chunks per channel
  
  
  // Follow the exact same pattern as input mapping
  // Tile order: column tiles OUTER, row blocks INNER (NW -> SW -> NE -> SE)
  int row_blocks_per_tile = M0 / ROWS_PER_BLOCK;  // blocks per row tile (8192)
  int numTilesM = ((M1 - 1) / M0) + 1;  // Number of row tiles
  int prev_row_tile = -1;
  
  for (int col_tile = 0; col_tile < N1; col_tile += N0) {
    int col_tile_idx = col_tile / N0;
    for (int block = 0; block < total_row_blocks; ++block) {
      // Calculate which row tile this block belongs to
      int row_tile_idx = block / row_blocks_per_tile;
      
      
      // Process each channel
      for (int ch = 0; ch < NUM_C_CH; ++ch) {
        // Process all rows for this channel (both first and second half)
        for (int local_row = 0; local_row < ROWS_PER_CHANNEL; ++local_row) {
          int cpu_row = block * ROWS_PER_BLOCK + ch * ROWS_PER_CHANNEL + local_row;
          
          for (int c = 0; c < N0; ++c) {
            int is_second_half = (local_row >= HALF_GROUP) ? 1 : 0;
            int base_row = local_row % HALF_GROUP; // Row index within its half (0-9)
            
            // Maps rows (0,10), (1,11), (2,12), etc. to consecutive chunks
            int chunk_idx = base_row;
            
            // Row-major tile order with packed allocation for partial tiles
            int row_tile_idx_calc = block / row_blocks_per_tile;
            int block_in_tile = block % row_blocks_per_tile;
            
            // Calculate actual blocks in this row tile (last tile may be partial)
            int blocks_in_this_tile;
            if (row_tile_idx_calc < numTilesM - 1) {
                blocks_in_this_tile = row_blocks_per_tile;  // Full tile
            } else {
                blocks_in_this_tile = total_row_blocks - row_tile_idx_calc * row_blocks_per_tile;  // Partial tile
            }
            
            // Base: all previous row tiles (each full tile has numTilesN col_tiles)
            int chunks_base = row_tile_idx_calc * numTilesN * row_blocks_per_tile * CHUNKS_PER_CHANNEL;
            
            // Add this col_tile's offset within this row_tile
            int chunks_per_col_in_this_tile = blocks_in_this_tile * CHUNKS_PER_CHANNEL;
            int local_chunk_idx = chunks_base + col_tile_idx * chunks_per_col_in_this_tile 
                                + block_in_tile * CHUNKS_PER_CHANNEL + chunk_idx;
            
            // Within each chunk, place first half in first row, second half in second row
            int offset_in_chunk = (is_second_half * N0) + c;
            int addr = local_chunk_idx * B_CHUNK_SIZE + offset_in_chunk;
            
            // Get values and compute error
            float fpga_val = fpga[ch][addr];
            float cpu_val = cpu[cpu_row][col_tile + c];
            
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

            // Debug: print first 10 mismatches (any difference)
            if (fpga_val != cpu_val && mismatch_count < MAX_MISMATCH_PRINT) {
              cout << fixed << setprecision(2);
              cout << "Mismatch #" << mismatch_count << ": "
                   << "CPU[" << cpu_row << "][" << (col_tile + c) << "]=" << cpu_val
                   << " vs FPGA[ch=" << ch << "][addr=" << addr << "]=" << fpga_val
                   << " | block=" << block << ", col_tile=" << col_tile 
                   << ", local_row=" << local_row << ", c=" << c
                   << " | chunk_idx=" << chunk_idx << ", local_chunk_idx=" << local_chunk_idx
                   << ", offset=" << offset_in_chunk
                   << " | relErr=" << relativeError << endl;
              cout << defaultfloat;  // Reset to default format
              mismatch_count++;
            }

            diffSum += diff;
            refSum += fabs(cpu_val);
            comparison_count++;
          }
        }
      }
    }
  }
  

  
  // clog << "Max Relative Error: " << maxRelativeError 
  //      << " CPU: " << max_cpu << " FPGA: " << max_fpga 
  //      << " at idx: " << maxRelativeErrorIdx << endl;

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

  clog << "Reading A" << filename << endl;

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

  vector<aligned_vector<uint64_t>> fpgaAmtx = prepareAmtx(tiledMatrices, numTilesM, numTilesK, M0, K0, M, K, nnz/* , USE_ROW_SHARE, USE_TREE_ADDER */);

  // Calculate total row blocks needed
  int total_row_blocks = (M1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  int row_blocks_per_tile = M0 / ROWS_PER_BLOCK;  // blocks per row tile (8192)

  const int CHUNKS_PER_CHANNEL = ROWS_PER_BLOCK/NUM_C_CH/2;
  cout << "M1: " << M1 << ", N1: " << N1 << ", NUM_C_CH: " << NUM_C_CH << endl;
  cout << "M0: " << M0 << ", row_blocks_per_tile: " << row_blocks_per_tile << endl;
  cout << "numTilesM: " << numTilesM << ", numTilesN: " << numTilesN << endl;
  cout << "fpgaCinMtx size: " << fpgaCinMtx[0].size() << " floats per channel" << endl;
  cout << "fpgaAmtx size: " << fpgaAmtx[0].size() << " uint64_t per channel" << endl;
  cout << "ROWS_PER_BLOCK: " << ROWS_PER_BLOCK << ", CHUNKS_PER_CHANNEL: " << CHUNKS_PER_CHANNEL << endl;
  cout << "B_CHUNK_SIZE: " << B_CHUNK_SIZE << endl;

  const int ROWS_PER_CHANNEL = ROWS_PER_BLOCK / NUM_C_CH; // 20 rows per channel
  const int HALF_GROUP = ROWS_PER_CHANNEL / 2; // 10 rows per half group

  // Tile order: column tiles OUTER, row blocks INNER
  for (int col_tile = 0; col_tile < N1; col_tile += N0) {
    int col_tile_idx = col_tile / N0;
    for (int block = 0; block < total_row_blocks; ++block) {
      // Process each channel's group of rows
      for (int ch = 0; ch < NUM_C_CH; ++ch) {
        // Process all rows for this channel (both first and second half)
        for (int local_row = 0; local_row < ROWS_PER_CHANNEL; ++local_row) {
          int cpu_row = block * ROWS_PER_BLOCK + ch * ROWS_PER_CHANNEL + local_row;
          
          if (cpu_row < M1) {
            for (int c = 0; c < N0; ++c) {
              float val = cpuCinMtx[cpu_row][col_tile + c];
              
              // Calculate exact address in interleaved pattern
              int is_second_half = (local_row >= HALF_GROUP) ? 1 : 0;
              int base_row = local_row % HALF_GROUP; // Row index within its half (0-9)
              
              // Maps rows (0,10), (1,11), (2,12), etc. to consecutive chunks
              int chunk_idx = base_row;
              
              // Row-major tile order with packed allocation for partial tiles
              int row_tile_idx = block / row_blocks_per_tile;
              int block_in_tile = block % row_blocks_per_tile;
              
              // Calculate actual blocks in this row tile (last tile may be partial)
              int blocks_in_this_tile;
              if (row_tile_idx < numTilesM - 1) {
                  blocks_in_this_tile = row_blocks_per_tile;  // Full tile
              } else {
                  blocks_in_this_tile = total_row_blocks - row_tile_idx * row_blocks_per_tile;  // Partial tile
              }
              
              // Base: all previous row tiles (each full tile has numTilesN col_tiles)
              int chunks_base = row_tile_idx * numTilesN * row_blocks_per_tile * CHUNKS_PER_CHANNEL;
              
              // Add this col_tile's offset within this row_tile
              int chunks_per_col_in_this_tile = blocks_in_this_tile * CHUNKS_PER_CHANNEL;
              int local_chunk_idx = chunks_base + col_tile_idx * chunks_per_col_in_this_tile 
                                  + block_in_tile * CHUNKS_PER_CHANNEL + chunk_idx;
              
              // Within each chunk, place first half in first row, second half in second row
              int offset_in_chunk = (is_second_half * N0) + c;
              int addr = local_chunk_idx * B_CHUNK_SIZE + offset_in_chunk;
              
              // Safety check and assignment
              if (ch >= NUM_C_CH || addr >= fpgaCinMtx[ch].size()) {
                cout << "Out of bounds access! ch=" << ch << ", addr=" << addr 
                     << ", max=" << fpgaCinMtx[ch].size() << endl;
                exit(1);
              }
              
              fpgaCinMtx[ch][addr] = val;
              
            }
          }
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
  if (precisionLoss == 0) {
    cout << "Success" << endl;
    return 0;
  }
  else {
    cout << "Failed:Precision Loss is not 0: " << precisionLoss << endl;
    return 1;
  }
}
