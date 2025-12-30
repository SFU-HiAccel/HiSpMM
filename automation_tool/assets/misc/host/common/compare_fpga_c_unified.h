// Unified fpgaCoutMtx (and fpgaCinMtx) comparison helpers.
//
// Goal:
// - Provide ONE host-only precision-loss function that matches the same addressing
//   layouts used for fpgaCinMtx/fpgaCoutMtx buffers.
//
// Usage:
// - Include this header ONLY from host code (g++ compilation), not kernel/HLS code.
// - Reuse the SAME `hispmm_host::CinPrepareConfig` you use for Cin packing.
//   (The address mapping for Cin and Cout is the same in these designs.)
//

#pragma once

#include <cmath>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "prepare_fpga_cin_unified.h"

namespace hispmm_host {

using std::invalid_argument;
using std::out_of_range;
using std::ostream;
using std::ostringstream;
using std::vector;

struct PrecisionLossStats {
  double diff_sum = 0.0;
  double ref_sum = 0.0;
  double max_relative_error = 0.0;
  float max_cpu = 0.0f;
  float max_fpga = 0.0f;
  int max_m = -1;
  int max_n = -1;
  int max_ch = -1;
  int64_t max_addr = -1;
};

struct PrecisionLossLog {
  // Optional CSV-style log of comparisons.
  // If set, emits rows like:
  //   idx,m,n,cpu_val,fpga_val,ch,addr,diff,rel_error
  ostream* out = nullptr;
  int* print_count = nullptr;
  int max_prints = 0;
};

static inline void LogCompare(const PrecisionLossLog& log,
                              int idx,
                              int m, int n,
                              float cpu_val, float fpga_val,
                              int ch, int64_t addr64,
                              double diff, double rel_err) {
  if (!log.out || !log.print_count || *log.print_count >= log.max_prints) return;
  (*log.out) << idx << ","
             << m << "," << n << ","
             << cpu_val << "," << fpga_val << ","
             << ch << "," << addr64 << ","
             << diff << "," << rel_err
             << "\n";
  (*log.print_count)++;
}

// Compute diffSum/refSum between CPU C matrix (row-major) and FPGA channelized buffers.
//
// - `addr_cfg` must describe the SAME layout used to store fpgaCoutMtx (and fpgaCinMtx).
// - `skip_oob=true` matches some existing debug implementations that simply skip
//   out-of-range addresses; set false to throw on any OOB (recommended for correctness).
inline double ComputePrecisionLossUnified(const vector<vector<float>>& cpu,
                                          const vector<aligned_vector<float>>& fpga,
                                          const CinPrepareConfig& addr_cfg,
                                          PrecisionLossStats* stats_out = nullptr,
                                          const PrecisionLossLog& log = {},
                                          bool skip_oob = true) {
  // Reuse the same validation as Cin preparation for common fields.
  ValidateCommon(addr_cfg, cpu, fpga);

  PrecisionLossStats st;
  int idx = 0;

  auto handle_sample = [&](int m, int n, int ch, int64_t addr64) {
    if (m < 0 || m >= addr_cfg.M1) return;
    if (n < 0 || n >= addr_cfg.N1) return;
    if (ch < 0 || ch >= addr_cfg.num_c_ch) return;

    if (addr64 < 0 || static_cast<uint64_t>(addr64) >= fpga[ch].size()) {
      if (skip_oob) return;
      ostringstream oss;
      oss << "ComputePrecisionLossUnified: out-of-bounds read: ch=" << ch << " addr=" << addr64
          << " max=" << fpga[ch].size() << " (m=" << m << " n=" << n << ")";
      throw out_of_range(oss.str());
    }

    const float cpu_val = cpu[m][n];
    const float fpga_val = fpga[ch][static_cast<size_t>(addr64)];
    const double diff = std::fabs(static_cast<double>(fpga_val) - static_cast<double>(cpu_val));
    const double ref = std::fabs(static_cast<double>(cpu_val));
    double rel_err = 0.0;
    if (cpu_val != 0.0f) {
      rel_err = diff / ref;
      if (rel_err > st.max_relative_error) {
        st.max_relative_error = rel_err;
        st.max_cpu = cpu_val;
        st.max_fpga = fpga_val;
        st.max_m = m;
        st.max_n = n;
        st.max_ch = ch;
        st.max_addr = addr64;
      }
    }

    st.diff_sum += diff;
    st.ref_sum += ref;
    LogCompare(log, idx, m, n, cpu_val, fpga_val, ch, addr64, diff, rel_err);
    idx++;
  };

  if (addr_cfg.layout == CinLayout::LinearChunkInterleave) {
    if (addr_cfg.m0 <= 0) throw invalid_argument("ComputePrecisionLossUnified(Linear): m0 must be > 0");
    int64_t linear_c_addr = 0;
    for (int i = 0; i < addr_cfg.M1; i += addr_cfg.m0) {
      for (int j = 0; j < addr_cfg.N1; j += addr_cfg.n0) {
        for (int ii = 0; (ii < addr_cfg.m0) && (i + ii < addr_cfg.M1); ii++) {
          for (int jj = 0; jj < addr_cfg.n0; jj++) {
            const int m = i + ii;
            const int n = j + jj;
            const int ch = static_cast<int>((linear_c_addr / addr_cfg.b_chunk_size) % addr_cfg.num_c_ch);
            const int64_t addr64 =
                (linear_c_addr / (static_cast<int64_t>(addr_cfg.num_c_ch) * addr_cfg.b_chunk_size)) *
                    addr_cfg.b_chunk_size +
                (linear_c_addr % addr_cfg.b_chunk_size);
            handle_sample(m, n, ch, addr64);
            linear_c_addr++;
          }
        }
      }
    }
  } else {
    // TiledPackedRows
    if (addr_cfg.total_row_blocks < 0 || addr_cfg.row_blocks_per_tile <= 0 || addr_cfg.num_tiles_m <= 0 ||
        addr_cfg.num_tiles_n <= 0 || addr_cfg.chunks_per_channel <= 0) {
      throw invalid_argument("ComputePrecisionLossUnified(Tiled): invalid tiling parameters");
    }

    const int rows_per_channel = addr_cfg.rows_per_block / addr_cfg.num_c_ch;
    if (rows_per_channel % 2 != 0) {
      throw invalid_argument("ComputePrecisionLossUnified(Tiled): rows_per_channel must be even");
    }
    if (addr_cfg.chunks_per_channel != rows_per_channel / 2) {
      throw invalid_argument("ComputePrecisionLossUnified(Tiled): chunks_per_channel must equal rows_per_channel/2");
    }
    const int half_group = rows_per_channel / 2;

    // Canonical order: col tiles outer, then row blocks.
    for (int col_tile = 0; col_tile < addr_cfg.N1; col_tile += addr_cfg.n0) {
      const int col_tile_idx = col_tile / addr_cfg.n0;
      for (int block = 0; block < addr_cfg.total_row_blocks; ++block) {
        const int row_tile_idx = block / addr_cfg.row_blocks_per_tile;
        const int block_in_tile = block % addr_cfg.row_blocks_per_tile;

        int blocks_in_this_tile = 0;
        if (row_tile_idx < addr_cfg.num_tiles_m - 1) {
          blocks_in_this_tile = addr_cfg.row_blocks_per_tile;
        } else {
          blocks_in_this_tile = addr_cfg.total_row_blocks - row_tile_idx * addr_cfg.row_blocks_per_tile;
        }
        if (blocks_in_this_tile <= 0) {
          throw invalid_argument("ComputePrecisionLossUnified(Tiled): computed blocks_in_this_tile <= 0");
        }

        const int64_t chunks_base = static_cast<int64_t>(row_tile_idx) * addr_cfg.num_tiles_n *
                                   addr_cfg.row_blocks_per_tile * addr_cfg.chunks_per_channel;
        const int64_t chunks_per_col_in_this_tile =
            static_cast<int64_t>(blocks_in_this_tile) * addr_cfg.chunks_per_channel;
        const int64_t col_tile_offset = static_cast<int64_t>(col_tile_idx) * chunks_per_col_in_this_tile;

        for (int ch = 0; ch < addr_cfg.num_c_ch; ++ch) {
          for (int local_row = 0; local_row < rows_per_channel; ++local_row) {
            const int m = block * addr_cfg.rows_per_block + ch * rows_per_channel + local_row;
            if (m >= addr_cfg.M1) continue;

            int chunk_idx = 0;
            int row_in_chunk = 0;
            if (addr_cfg.pairing == RowPairing::AdjacentPair) {
              chunk_idx = local_row / 2;
              row_in_chunk = local_row % 2;
            } else {  // HalfGroupPair
              row_in_chunk = (local_row >= half_group) ? 1 : 0;
              chunk_idx = local_row % half_group;
            }

            const int64_t local_chunk_idx =
                chunks_base + col_tile_offset +
                static_cast<int64_t>(block_in_tile) * addr_cfg.chunks_per_channel +
                static_cast<int64_t>(chunk_idx);

            for (int c = 0; c < addr_cfg.n0; ++c) {
              const int n = col_tile + c;
              const int offset_in_chunk = row_in_chunk * addr_cfg.n0 + c;
              const int64_t addr64 =
                  local_chunk_idx * static_cast<int64_t>(addr_cfg.b_chunk_size) + offset_in_chunk;
              handle_sample(m, n, ch, addr64);
            }
          }
        }
      }
    }
  }

  if (stats_out) *stats_out = st;
  if (st.ref_sum == 0.0) return 0.0;
  return st.diff_sum / st.ref_sum;
}

}  // namespace hispmm_host


