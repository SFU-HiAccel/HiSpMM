// Unified FPGA A-matrix (fpgaAmtx) preparation helpers.
//
// Goal:
// - Provide ONE host-side implementation for preparing fpgaAmtx across:
//   - non-row-sharing designs (only prepareAmtx1)
//   - row-sharing-capable designs (prepareAmtx1/prepareAmtx2 + selection policy)
//
// Notes / constraints captured from user:
// - Some kernels are NOT capable of row sharing (no DRDN); those must use prepareAmtx1 only.
// - Some kernels ARE capable; support both prepareAmtx1 and prepareAmtx2.
// - Even if a kernel is capable, kernel-selection may choose to disable row sharing.
//
// Integration intent:
// - Copy this file + its .cpp into a host build and compile it with the host sources.
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tapa.h>

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

struct CSRMatrix {
  std::vector<int> row_offsets;
  std::vector<int> col_indices;
  std::vector<float> values;
};

struct rcv {
  int r;
  int c;
  float v;
};

// Matrix IO + basic helpers (implemented in `prepare_amt_unified.cpp`).
void readMatrixCSC(char* filename,
                   std::vector<float>& values,
                   std::vector<int>& rowIndices,
                   std::vector<int>& colOffsets,
                   int& rows,
                   int& cols,
                   int& nnz);

void convertCSCtoCSR(const std::vector<float>& cscValues,
                     const std::vector<int>& cscRowIndices,
                     const std::vector<int>& cscColOffsets,
                     std::vector<float>& csrValues,
                     std::vector<int>& csrColIndices,
                     std::vector<int>& csrRowOffsets,
                     int rows,
                     int cols,
                     int nnz);

void printMatrixCSR(std::vector<float> values,
                    std::vector<int> columns,
                    std::vector<int> rowPtr,
                    int numRows,
                    int numCols);

std::vector<std::vector<CSRMatrix>> tileCSRMatrix(const CSRMatrix& originalMatrix,
                                                  int numRows,
                                                  int numCols,
                                                  int tileRows,
                                                  int tileCols,
                                                  int numTilesRows,
                                                  int numTilesCols);

// Legacy packet format helpers (kept identical to the old helper header).
inline uint64_t encode(bool tileEnd, bool rowEnd, bool sharedRow, uint16_t row, int col, uint32_t val) {
  uint64_t res = 0;  // 16bits
  res |= row;
  res <<= 1;
  res |= tileEnd;  // will be 47th bit
  res <<= 1;
  res |= rowEnd;  // will be 46th bit
  res <<= 1;
  res |= sharedRow;  // will be 45th bit
  res <<= 13;
  res |= col & (0x1FFF);  // 13 bits col
  res <<= 32;
  res |= val;  // 32 bits val
  return res;
}

inline void decode(uint64_t a,
                   bool& tileEnd,
                   bool& rowEnd,
                   bool& sharedRow,
                   uint16_t& row16,
                   uint16_t& col,
                   float& val) {
  uint32_t cval = a & 0xFFFFFFFF;
  a >>= 32;
  col = a & 0x1FFF;
  a >>= 13;
  sharedRow = a & 1;
  a >>= 1;
  rowEnd = a & 1;
  a >>= 1;
  tileEnd = a & 1;
  a >>= 1;
  row16 = a & 0xFFFF;
  val = *(float*)&cval;
}

// Legacy signature retained for compatibility with older host code.
// NOTE: This wrapper ignores the obsolete `Window` argument; the unified path
// uses `Depth` as the tile row count (`tile_depth`).
std::vector<aligned_vector<uint64_t>> prepareAmtx(std::vector<std::vector<CSRMatrix>> tiledMatrices,
                                                  const int numTilesRows,
                                                  const int numTilesCols,
                                                  const int Depth,
                                                  const int Window,
                                                  const int rows,
                                                  const int cols,
                                                  const int nnz);

namespace hispmm_host {

enum class RowSharingPolicy {
  // Never use row sharing, even if the kernel is capable.
  kForceDisabled,
  // Always use row sharing (requires capability=true).
  kForceEnabled,
  // Choose based on a heuristic (delta-improvement by default).
  kAuto,
};

struct PrepareAmtxConfig {
  // Hardware capability (from kernel configuration / DSE decision):
  // - false: row-sharing not supported (no DRDN) => always use prepareAmtx1.
  // - true:  row-sharing supported => can use prepareAmtx2 depending on policy.
  // Codegen/DSE should set this per-kernel.
  //
  // Default is conservative (disabled) to avoid accidentally enabling row sharing
  // for kernels that cannot accommodate DRDN.
  bool kernel_supports_row_sharing = false;

  // User / DSE override (from kernel selection or CLI):
  RowSharingPolicy row_sharing = RowSharingPolicy::kAuto;

  // Greedy shared-row selection limit per tile.
  // If <= 0, defaults to tile_depth / 2 (matching existing code).
  int shared_row_limit = -1;

  // Auto decision threshold: if delta improvement >= threshold, enable row sharing.
  // (delta improvement is (delta_no_rs - delta_rs) / delta_no_rs * 100).
  double delta_improvement_threshold_percent = 25.0;

  // Print summary messages (decision + deltas).
  bool print_summary = false;
};

struct PrepareAmtxResult {
  std::vector<aligned_vector<uint64_t>> fpgaAmtx;

  // Which mode was actually used.
  bool used_row_sharing = false;

  // Run length in "rows" consumed by the kernel's MM2S_A path.
  // Matches common host usage: fpgaAmtx[0].size() / PES_PER_CH.
  int run_len = 0;

  // Useful diagnostics for Auto mode.
  double delta_no_row_sharing = 0.0;
  double delta_with_row_sharing = 0.0;
  double delta_improvement_percent = 0.0;
};

// Prepare fpgaAmtx for a tiled CSR matrix.
//
// Parameters:
// - tiledMatrices: [numTilesRows][numTilesCols] tiles in CSR.
// - tile_depth: number of rows per tile (Depth in existing code; typically M0).
// - nnz_total: total nnz (used only for compatibility / potential prints).
PrepareAmtxResult PrepareAmtxUnified(const std::vector<std::vector<CSRMatrix>>& tiledMatrices,
                                     int numTilesRows,
                                     int numTilesCols,
                                     int tile_depth,
                                     int nnz_total,
                                     const PrepareAmtxConfig& cfg);

}  // namespace hispmm_host


