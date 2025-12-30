// Unified fpgaCinMtx preparation helpers.
//
// This file is intentionally standalone and does not modify any existing host code.
// You can copy the function(s) you need into a host file, or compile/link this file
// alongside your host (as long as it sees the same `aligned_vector` typedef and
// uses compatible N0/B_CHUNK_SIZE constants).
//
// Supported layouts:
// - LinearChunkInterleave: the classic "linear_c_addr" interleave used by A4 and A6 hosts.
// - TiledPackedRows: the tiled/block packed layout used by A8C8, A8C4, A10C4, etc.
//   This supports two row pairing modes:
//     * AdjacentPair: pairs rows (0,1), (2,3), ...
//     * HalfGroupPair: pairs rows (0,half), (1,half+1), ... (A10C4 style)
//
// Notes:
// - "tiled packed" here refers to the addressing scheme that matches the kernel's
//   MM2S_C/S2MM_C traversal over (tileM,tileN) and the arbiter's 2-row packing into
//   float_vB (B_CHUNK_SIZE == 2*N0 floats).
#pragma once
#include <cstdint>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Expect these to be available from the project's host codebase.
// - aligned_vector<float> (typically `tapa::aligned_allocator`)
// - `N0`, `B_CHUNK_SIZE` if you want to omit passing them explicitly
// If you prefer fully-standalone code, change aligned_vector to `std::vector<float>`.
#include "prepare_amt_unified.h"

namespace hispmm_host {

using std::invalid_argument;
using std::out_of_range;
using std::ostream;
using std::ostringstream;
using std::vector;

enum class CinLayout {
  // A4-style: linear address increases over tiles, interleaved across channels by chunk.
  LinearChunkInterleave,
  // A8/A10-style: tile/block packed addressing, 2-row packed chunks.
  TiledPackedRows,
};

enum class RowPairing {
  // Pack consecutive rows into one chunk: (0,1), (2,3), ...
  AdjacentPair,
  // Pack rows separated by half-group: (0,half), (1,half+1), ... (A10C4 style)
  HalfGroupPair,
};

struct CinPrepareConfig {
  CinLayout layout = CinLayout::TiledPackedRows;
  RowPairing pairing = RowPairing::AdjacentPair;

  // Matrix dims (padded).
  int M1 = 0;
  int N1 = 0;

  // Hardware/padding constants.
  int num_c_ch = 0;
  int rows_per_block = 0;   // typically NUM_PES
  int n0 = 0;               // typically N0
  int b_chunk_size = 0;     // typically 2*n0 (B_CHUNK_SIZE)

  // Tiling params (only for TiledPackedRows).
  int total_row_blocks = 0;
  int row_blocks_per_tile = 0;
  int num_tiles_m = 0;
  int num_tiles_n = 0;
  int chunks_per_channel = 0;  // rows_per_channel/2

  // Linear interleave params (only for LinearChunkInterleave).
  int m0 = 0;  // tile height
};

struct CinMappingLog {
  // Optional CSV-style log of mappings.
  // If set, we will emit rows like: fpgaCinMtx[ch][addr],cpuCinMtx[m][n],val
  ostream* out = nullptr;
  int* print_count = nullptr;
  int max_prints = 0;
};

static inline void ValidateCommon(const CinPrepareConfig& cfg,
                                  const vector<vector<float>>& cpuCinMtx,
                                  const vector<aligned_vector<float>>& fpgaCinMtx) {
  if (cfg.M1 < 0 || cfg.N1 < 0) throw invalid_argument("PrepareFpgaCinMtx: M1/N1 must be non-negative");
  if (cfg.num_c_ch <= 0) throw invalid_argument("PrepareFpgaCinMtx: num_c_ch must be > 0");
  if (cfg.rows_per_block <= 0) throw invalid_argument("PrepareFpgaCinMtx: rows_per_block must be > 0");
  if (cfg.n0 <= 0) throw invalid_argument("PrepareFpgaCinMtx: n0 must be > 0");
  if (cfg.b_chunk_size <= 0) throw invalid_argument("PrepareFpgaCinMtx: b_chunk_size must be > 0");
  if (cfg.b_chunk_size != 2 * cfg.n0) {
    throw invalid_argument("PrepareFpgaCinMtx: expected b_chunk_size == 2*n0");
  }

  if (static_cast<int>(cpuCinMtx.size()) < cfg.M1) {
    throw invalid_argument("PrepareFpgaCinMtx: cpuCinMtx has fewer rows than M1");
  }
  if (static_cast<int>(fpgaCinMtx.size()) < cfg.num_c_ch) {
    throw invalid_argument("PrepareFpgaCinMtx: fpgaCinMtx has fewer channels than num_c_ch");
  }
  if (cfg.N1 % cfg.n0 != 0) {
    throw invalid_argument("PrepareFpgaCinMtx: N1 must be divisible by n0");
  }
  if (cfg.rows_per_block % cfg.num_c_ch != 0) {
    throw invalid_argument("PrepareFpgaCinMtx: rows_per_block must be divisible by num_c_ch");
  }
}

static inline void LogMap(const CinMappingLog& log,
                          int ch, int64_t addr64,
                          int m, int n, float val) {
  if (!log.out || !log.print_count || *log.print_count >= log.max_prints) return;
  (*log.out) << "fpgaCinMtx[" << ch << "][" << addr64 << "],"
             << "cpuCinMtx[" << m << "][" << n << "],"
             << val << "\n";
  (*log.print_count)++;
}

// Unified entry point.
//
// - cpuCinMtx: CPU-side Cin, shape [M1][N1] (row-major)
// - fpgaCinMtx: per-channel flat buffers (size must match your host allocation)
//
// This function throws std::invalid_argument/std::out_of_range on misconfig/OOB.
inline void PrepareFpgaCinMtx(const vector<vector<float>>& cpuCinMtx,
                              vector<aligned_vector<float>>& fpgaCinMtx,
                              const CinPrepareConfig& cfg,
                              const CinMappingLog& log = {}) {
  ValidateCommon(cfg, cpuCinMtx, fpgaCinMtx);

  if (cfg.layout == CinLayout::LinearChunkInterleave) {
    if (cfg.m0 <= 0) throw invalid_argument("PrepareFpgaCinMtx(Linear): m0 must be > 0");
    int64_t linear_c_addr = 0;
    for (int i = 0; i < cfg.M1; i += cfg.m0) {
      for (int j = 0; j < cfg.N1; j += cfg.n0) {
        for (int ii = 0; (ii < cfg.m0) && (i + ii < cfg.M1); ii++) {
          for (int jj = 0; jj < cfg.n0; jj++) {
            const int m = i + ii;
            const int n = j + jj;
            const int ch = static_cast<int>((linear_c_addr / cfg.b_chunk_size) % cfg.num_c_ch);
            const int64_t addr64 =
                (linear_c_addr / (static_cast<int64_t>(cfg.num_c_ch) * cfg.b_chunk_size)) * cfg.b_chunk_size +
                (linear_c_addr % cfg.b_chunk_size);

            if (ch < 0 || ch >= cfg.num_c_ch) throw out_of_range("PrepareFpgaCinMtx(Linear): ch out of range");
            if (addr64 < 0 || static_cast<uint64_t>(addr64) >= fpgaCinMtx[ch].size()) {
              ostringstream oss;
              oss << "PrepareFpgaCinMtx(Linear): out-of-bounds write: ch=" << ch << " addr=" << addr64
                  << " max=" << fpgaCinMtx[ch].size()
                  << " (m=" << m << " n=" << n << " linear=" << linear_c_addr << ")";
              throw out_of_range(oss.str());
            }

            const float val = cpuCinMtx[m][n];
            fpgaCinMtx[ch][static_cast<size_t>(addr64)] = val;
            LogMap(log, ch, addr64, m, n, val);
            linear_c_addr++;
          }
        }
      }
    }
    return;
  }

  // TiledPackedRows
  if (cfg.total_row_blocks < 0 || cfg.row_blocks_per_tile <= 0 || cfg.num_tiles_m <= 0 || cfg.num_tiles_n <= 0 ||
      cfg.chunks_per_channel <= 0) {
    throw invalid_argument("PrepareFpgaCinMtx(Tiled): invalid tiling parameters");
  }

  const int rows_per_channel = cfg.rows_per_block / cfg.num_c_ch;
  if (rows_per_channel % 2 != 0) throw invalid_argument("PrepareFpgaCinMtx(Tiled): rows_per_channel must be even");
  if (cfg.chunks_per_channel != rows_per_channel / 2) {
    throw invalid_argument("PrepareFpgaCinMtx(Tiled): chunks_per_channel must equal rows_per_channel/2");
  }
  const int half_group = rows_per_channel / 2;

  // Canonical order (safe for memory correctness): col tiles outer, then row blocks.
  for (int col_tile = 0; col_tile < cfg.N1; col_tile += cfg.n0) {
    const int col_tile_idx = col_tile / cfg.n0;
    for (int block = 0; block < cfg.total_row_blocks; ++block) {
      const int row_tile_idx = block / cfg.row_blocks_per_tile;
      const int block_in_tile = block % cfg.row_blocks_per_tile;

      int blocks_in_this_tile = 0;
      if (row_tile_idx < cfg.num_tiles_m - 1) {
        blocks_in_this_tile = cfg.row_blocks_per_tile;
      } else {
        blocks_in_this_tile = cfg.total_row_blocks - row_tile_idx * cfg.row_blocks_per_tile;
      }
      if (blocks_in_this_tile <= 0) {
        throw invalid_argument("PrepareFpgaCinMtx(Tiled): computed blocks_in_this_tile <= 0");
      }

      const int64_t chunks_base =
          static_cast<int64_t>(row_tile_idx) * cfg.num_tiles_n * cfg.row_blocks_per_tile * cfg.chunks_per_channel;
      const int64_t chunks_per_col_in_this_tile =
          static_cast<int64_t>(blocks_in_this_tile) * cfg.chunks_per_channel;
      const int64_t col_tile_offset = static_cast<int64_t>(col_tile_idx) * chunks_per_col_in_this_tile;

      for (int ch = 0; ch < cfg.num_c_ch; ++ch) {
        for (int local_row = 0; local_row < rows_per_channel; ++local_row) {
          const int m = block * cfg.rows_per_block + ch * rows_per_channel + local_row;
          if (m >= cfg.M1) continue;
          if (static_cast<int>(cpuCinMtx[m].size()) < cfg.N1) {
            throw invalid_argument("PrepareFpgaCinMtx(Tiled): cpuCinMtx row has fewer columns than N1");
          }

          int chunk_idx = 0;
          int row_in_chunk = 0;
          if (cfg.pairing == RowPairing::AdjacentPair) {
            chunk_idx = local_row / 2;
            row_in_chunk = local_row % 2;
          } else {  // HalfGroupPair
            row_in_chunk = (local_row >= half_group) ? 1 : 0;
            chunk_idx = local_row % half_group;
          }

          const int64_t local_chunk_idx =
              chunks_base + col_tile_offset +
              static_cast<int64_t>(block_in_tile) * cfg.chunks_per_channel +
              static_cast<int64_t>(chunk_idx);

          for (int c = 0; c < cfg.n0; ++c) {
            const int n = col_tile + c;
            const float val = cpuCinMtx[m][n];

            const int offset_in_chunk = row_in_chunk * cfg.n0 + c;
            const int64_t addr64 = local_chunk_idx * static_cast<int64_t>(cfg.b_chunk_size) + offset_in_chunk;

            if (addr64 < 0 || static_cast<uint64_t>(addr64) >= fpgaCinMtx[ch].size()) {
              ostringstream oss;
              oss << "PrepareFpgaCinMtx(Tiled): out-of-bounds write: ch=" << ch << " addr=" << addr64
                  << " max=" << fpgaCinMtx[ch].size()
                  << " (m=" << m << " n=" << n << " block=" << block << " col_tile=" << col_tile
                  << " local_row=" << local_row << " chunk_idx=" << chunk_idx << ")";
              throw out_of_range(oss.str());
            }
            fpgaCinMtx[ch][static_cast<size_t>(addr64)] = val;
            LogMap(log, ch, addr64, m, n, val);
          }
        }
      }
    }
  }
}

}  // namespace hispmm_host


