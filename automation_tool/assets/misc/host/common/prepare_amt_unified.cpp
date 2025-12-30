#include "prepare_amt_unified.h"

#include "mmio.h"   // MatrixMarket reader types/functions.
#include "hispmm.h"   // NUM_* macros used by A-matrix packing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hispmm_host {
namespace {

using std::vector;

struct TileWorkStats {
  // Tile sizes (in units matching existing code: (max_lane_load + PADDING) * II_DIST)
  vector<vector<int>> tile_sizes;
  int total_size = 0;   // sum(tile_sizes)
  double delta = 0.0;   // stddev/mean (per-PE summed load)
};

static inline int CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

static inline void ValidateInputs(const vector<vector<CSRMatrix>>& tiledMatrices,
                                  int numTilesRows,
                                  int numTilesCols,
                                  int tile_depth) {
  if (numTilesRows <= 0 || numTilesCols <= 0) {
    throw std::invalid_argument("PrepareAmtxUnified: numTilesRows/numTilesCols must be > 0");
  }
  if (tile_depth <= 0) {
    throw std::invalid_argument("PrepareAmtxUnified: tile_depth must be > 0");
  }
  if (static_cast<int>(tiledMatrices.size()) != numTilesRows) {
    throw std::invalid_argument("PrepareAmtxUnified: tiledMatrices.size() != numTilesRows");
  }
  for (int i = 0; i < numTilesRows; ++i) {
    if (static_cast<int>(tiledMatrices[i].size()) != numTilesCols) {
      throw std::invalid_argument("PrepareAmtxUnified: tiledMatrices[i].size() != numTilesCols");
    }
  }
}

// Greedy shared-row selection per tile (ported from existing helper_functions.cpp).
//
// Output:
// - num_shared_rows[i][j] = how many rows are shared in tile (i,j)
// - shared_rows[i][j] = list of shared row ids (sorted ascending)
static void ComputeSharedRowsGreedy(const vector<vector<CSRMatrix>>& tiledMatrices,
                                    int numTilesRows,
                                    int numTilesCols,
                                    int tile_depth,
                                    int shared_row_limit,
                                    vector<vector<int>>* num_shared_rows,
                                    vector<vector<vector<int>>>* shared_rows) {
  num_shared_rows->assign(numTilesRows, vector<int>(numTilesCols, 0));
  shared_rows->assign(numTilesRows, vector<vector<int>>(numTilesCols, vector<int>{}));

  if (shared_row_limit <= 0) return;

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      const CSRMatrix& tile = tiledMatrices[i][j];
      const int num_rows = static_cast<int>(tile.row_offsets.size()) - 1;
      const int depth = std::min(tile_depth, num_rows);

      vector<vector<int>> rowCounts(depth, vector<int>(2, 0));
      vector<vector<int>> peWorkloads(NUM_PES, vector<int>(2, 0));

      for (int r = 0; r < depth; ++r) {
        rowCounts[r][0] = r;
        rowCounts[r][1] = tile.row_offsets[r + 1] - tile.row_offsets[r];
        peWorkloads[r % NUM_PES][0] = r % NUM_PES;
        peWorkloads[r % NUM_PES][1] += rowCounts[r][1];
      }

      std::sort(rowCounts.begin(), rowCounts.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      int maxVal = 0;
      for (int p = 0; p < NUM_PES; ++p) maxVal = std::max(maxVal, peWorkloads[p][1]);

      const int total = tile.row_offsets[depth];
      if (total <= 0) {
        // Empty tile: nothing to share.
        (*num_shared_rows)[i][j] = 0;
        (*shared_rows)[i][j].clear();
        continue;
      }

      int scheduled = (maxVal + PADDING) * NUM_PES;
      float imb = static_cast<float>(scheduled - total) / static_cast<float>(total);

      int extraCycles = 0;
      vector<int> removedRows;
      removedRows.reserve(shared_row_limit);

      const int limit = std::min(shared_row_limit, depth);
      for (int ii = 0; ii < limit; ++ii) {
        if (maxVal < 2) break;

        const int row_id = rowCounts[ii][0];
        const int row_len = rowCounts[ii][1];
        const int newTotal = total - row_len;
        if (newTotal <= 0) break;

        vector<vector<int>> newPeWorkloads = peWorkloads;
        int newMaxVal = 0;
        for (int p = 0; p < NUM_PES; ++p) {
          if (newPeWorkloads[p][0] == (row_id % NUM_PES)) newPeWorkloads[p][1] -= row_len;
          newMaxVal = std::max(newMaxVal, newPeWorkloads[p][1]);
        }

        scheduled = NUM_PES * newMaxVal;
        const float new_imb = static_cast<float>(scheduled - newTotal) / static_cast<float>(newTotal);

        // Remove if imbalance improves, or keep first two rows as in legacy code.
        if (((imb - new_imb) > 0.0f) || (ii < 2)) {
          peWorkloads = std::move(newPeWorkloads);
          maxVal = newMaxVal;
          extraCycles += CeilDiv(std::max(0, row_len - 1), NUM_PES);
          removedRows.push_back(row_id);
        }

        if ((std::fabs(imb - new_imb) <= 0.01f) && (new_imb < 2.0f)) break;
        imb = new_imb;
      }

      std::sort(removedRows.begin(), removedRows.end());
      (*num_shared_rows)[i][j] = static_cast<int>(removedRows.size());
      (*shared_rows)[i][j] = std::move(removedRows);
      (void)extraCycles;  // retained for parity with legacy logic; not used by unified path.
    }
  }
}

static TileWorkStats ComputePEloads_NoRowSharing(const vector<vector<CSRMatrix>>& tiledMatrices,
                                                 int numTilesRows,
                                                 int numTilesCols,
                                                 int tile_depth) {
  TileWorkStats out;
  out.tile_sizes.assign(numTilesRows, vector<int>(numTilesCols, 0));

  double load_sum = 0.0;
  double load_sq_sum = 0.0;
  long long load_count = 0;

  #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:load_sum, load_sq_sum, load_count)
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      const CSRMatrix& tile = tiledMatrices[i][j];
      const int num_rows = static_cast<int>(tile.row_offsets.size()) - 1;
      const int depth = std::min(tile_depth, num_rows);

      vector<vector<int>> Loads(NUM_PES, vector<int>(II_DIST, 0));
      vector<vector<int>> sorted_rows(depth, vector<int>(2, 0));
      for (int r = 0; r < depth; ++r) {
        sorted_rows[r][0] = r;
        sorted_rows[r][1] = tile.row_offsets[r + 1] - tile.row_offsets[r];
      }
      std::sort(sorted_rows.begin(), sorted_rows.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      for (int rr = 0; rr < depth; ++rr) {
        const int rowSize = sorted_rows[rr][1];
        const int pe = sorted_rows[rr][0] % NUM_PES;
        int min_idx = 0;
        int min_val = Loads[pe][0];
        for (int ii = 1; ii < II_DIST; ++ii) {
          if (Loads[pe][ii] < min_val) {
            min_val = Loads[pe][ii];
            min_idx = ii;
          }
        }
        Loads[pe][min_idx] += rowSize;
      }

      int max_lane_load = 0;
      for (int p = 0; p < NUM_PES; ++p) {
        int pe_load = 0;
        for (int ii = 0; ii < II_DIST; ++ii) {
          max_lane_load = std::max(max_lane_load, Loads[p][ii]);
          pe_load += Loads[p][ii];
        }
        const double load = static_cast<double>(pe_load);
        load_sum += load;
        load_sq_sum += load * load;
        load_count += 1;
      }

      out.tile_sizes[i][j] = (max_lane_load + PADDING) * II_DIST;
    }
  }

  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) out.total_size += out.tile_sizes[i][j];
  }

  if (load_count > 0) {
    const double mean = load_sum / static_cast<double>(load_count);
    double var = load_sq_sum / static_cast<double>(load_count) - mean * mean;
    if (var < 0.0) var = 0.0;
    const double stddev = std::sqrt(var);
    out.delta = (mean != 0.0) ? (stddev / mean) : 0.0;
  }
  return out;
}

static TileWorkStats ComputePEloads_WithRowSharing(const vector<vector<CSRMatrix>>& tiledMatrices,
                                                   int numTilesRows,
                                                   int numTilesCols,
                                                   int tile_depth,
                                                   const vector<vector<int>>& numSharedRows,
                                                   const vector<vector<vector<int>>>& sharedRows) {
  TileWorkStats out;
  out.tile_sizes.assign(numTilesRows, vector<int>(numTilesCols, 0));

  double load_sum = 0.0;
  double load_sq_sum = 0.0;
  long long load_count = 0;

  #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:load_sum, load_sq_sum, load_count)
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      const CSRMatrix& tile = tiledMatrices[i][j];
      const int num_rows = static_cast<int>(tile.row_offsets.size()) - 1;
      const int depth = std::min(tile_depth, num_rows);

      vector<vector<int>> Loads(NUM_PES, vector<int>(II_DIST, 0));

      // Shared rows first (each shared row contributes ceil(row_size/NUM_PES) to each PE).
      const int num_shared = numSharedRows[i][j];
      vector<vector<int>> sortedShared(num_shared, vector<int>(2, 0));
      for (int k = 0; k < num_shared; ++k) {
        const int row_id = sharedRows[i][j][k];
        sortedShared[k][0] = row_id;
        sortedShared[k][1] = tile.row_offsets[row_id + 1] - tile.row_offsets[row_id];
      }
      std::sort(sortedShared.begin(), sortedShared.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      vector<char> is_shared(depth, 0);
      for (int k = 0; k < num_shared; ++k) {
        const int row_id = sharedRows[i][j][k];
        if (0 <= row_id && row_id < depth) is_shared[row_id] = 1;
      }

      for (int k = 0; k < num_shared; ++k) {
        const int row_size = sortedShared[k][1];
        const int load_size = CeilDiv(row_size, NUM_PES);
        for (int pe = 0; pe < NUM_PES; ++pe) {
          int min_idx = 0;
          int min_val = Loads[pe][0];
          for (int ii = 1; ii < II_DIST; ++ii) {
            if (Loads[pe][ii] < min_val) {
              min_val = Loads[pe][ii];
              min_idx = ii;
            }
          }
          Loads[pe][min_idx] += load_size;
        }
      }

      // Remaining rows: like no-row-sharing scheduling.
      vector<vector<int>> rem_rows;
      rem_rows.reserve(depth);
      for (int r = 0; r < depth; ++r) {
        if (is_shared[r]) continue;
        const int row_size = tile.row_offsets[r + 1] - tile.row_offsets[r];
        rem_rows.push_back({r, row_size});
      }
      std::sort(rem_rows.begin(), rem_rows.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      for (const auto& rr : rem_rows) {
        const int row_size = rr[1];
        const int pe = rr[0] % NUM_PES;
        int min_idx = 0;
        int min_val = Loads[pe][0];
        for (int ii = 1; ii < II_DIST; ++ii) {
          if (Loads[pe][ii] < min_val) {
            min_val = Loads[pe][ii];
            min_idx = ii;
          }
        }
        Loads[pe][min_idx] += row_size;
      }

      int max_lane_load = 0;
      for (int p = 0; p < NUM_PES; ++p) {
        int pe_load = 0;
        for (int ii = 0; ii < II_DIST; ++ii) {
          max_lane_load = std::max(max_lane_load, Loads[p][ii]);
          pe_load += Loads[p][ii];
        }
        const double load = static_cast<double>(pe_load);
        load_sum += load;
        load_sq_sum += load * load;
        load_count += 1;
      }

      out.tile_sizes[i][j] = (max_lane_load + PADDING) * II_DIST;
    }
  }

  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) out.total_size += out.tile_sizes[i][j];
  }

  if (load_count > 0) {
    const double mean = load_sum / static_cast<double>(load_count);
    double var = load_sq_sum / static_cast<double>(load_count) - mean * mean;
    if (var < 0.0) var = 0.0;
    const double stddev = std::sqrt(var);
    out.delta = (mean != 0.0) ? (stddev / mean) : 0.0;
  }
  return out;
}

static vector<aligned_vector<uint64_t>> PackAmtx_NoRowSharing(const vector<vector<CSRMatrix>>& tiledMatrices,
                                                              int numTilesRows,
                                                              int numTilesCols,
                                                              const vector<vector<int>>& tileSizes) {
  vector<vector<int>> tileOffsets(numTilesRows, vector<int>(numTilesCols, 0));
  int totalSize = 0;
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      tileOffsets[i][j] = PES_PER_CH * totalSize;
      totalSize += tileSizes[i][j];
    }
  }

  vector<aligned_vector<uint64_t>> fpgaAmtx(NUM_A_CH, aligned_vector<uint64_t>(PES_PER_CH * totalSize));

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      vector<vector<int>> Loads(NUM_PES, vector<int>(II_DIST, 0));
      const int curr_tile_offset = tileOffsets[i][j];
      const int curr_tile_size = tileSizes[i][j];
      const CSRMatrix& tile = tiledMatrices[i][j];
      const int num_rows = static_cast<int>(tile.row_offsets.size()) - 1;

      vector<vector<int>> sorted_rows(num_rows, vector<int>(2, 0));
      for (int r = 0; r < num_rows; ++r) {
        sorted_rows[r][0] = r;
        sorted_rows[r][1] = tile.row_offsets[r + 1] - tile.row_offsets[r];
      }
      std::sort(sorted_rows.begin(), sorted_rows.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      for (int rr = 0; rr < num_rows; ++rr) {
        const int row_no = sorted_rows[rr][0];
        const int pe = row_no % NUM_PES;

        int min_idx = 0;
        int min_val = Loads[pe][0];
        for (int ii = 1; ii < II_DIST; ++ii) {
          if (Loads[pe][ii] < min_val) {
            min_val = Loads[pe][ii];
            min_idx = ii;
          }
        }

        const int ch_no = pe / PES_PER_CH;
        const int inter_ch_pe = pe % PES_PER_CH;
        const uint16_t row16 = static_cast<uint16_t>(row_no / NUM_PES);

        for (int ind = tile.row_offsets[row_no]; ind < tile.row_offsets[row_no + 1]; ++ind) {
          const int col_id = tile.col_indices[ind];
          const float value = tile.values[ind];
          const uint32_t val_bits = *reinterpret_cast<const uint32_t*>(&value);

          const int addr = curr_tile_offset + (((Loads[pe][min_idx] * II_DIST) + min_idx) * PES_PER_CH) + inter_ch_pe;
          fpgaAmtx[ch_no][addr] = encode(false, true, false, row16, col_id, val_bits);
          Loads[pe][min_idx]++;
        }
      }

      // Padding to equalize lane lengths and mark tile end.
      for (int p = 0; p < NUM_PES; ++p) {
        const int ch_no = p / PES_PER_CH;
        const int inter_ch_pe = p % PES_PER_CH;
        for (int ii = 0; ii < II_DIST; ++ii) {
          while (Loads[p][ii] < (curr_tile_size / II_DIST)) {
            const bool tileEnd = (Loads[p][ii] == (curr_tile_size / II_DIST) - 1) && (ii == II_DIST - 1);
            const int col_id = 0;
            const uint16_t row16 = 0;
            const float value = 0.0f;
            const uint32_t val_bits = *reinterpret_cast<const uint32_t*>(&value);

            const int addr = curr_tile_offset + ((Loads[p][ii]++) * II_DIST + ii) * PES_PER_CH + inter_ch_pe;
            fpgaAmtx[ch_no][addr] = encode(tileEnd, false, false, row16, col_id, val_bits);
          }
        }
      }
    }
  }

  return fpgaAmtx;
}

static vector<aligned_vector<uint64_t>> PackAmtx_WithRowSharing(const vector<vector<CSRMatrix>>& tiledMatrices,
                                                                int numTilesRows,
                                                                int numTilesCols,
                                                                const vector<vector<int>>& numSharedRows,
                                                                const vector<vector<vector<int>>>& sharedRows,
                                                                const vector<vector<int>>& tileSizes) {
  vector<vector<int>> tileOffsets(numTilesRows, vector<int>(numTilesCols, 0));
  int totalSize = 0;
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      tileOffsets[i][j] = PES_PER_CH * totalSize;
      totalSize += tileSizes[i][j];
    }
  }

  vector<aligned_vector<uint64_t>> fpgaAmtx(NUM_A_CH, aligned_vector<uint64_t>(PES_PER_CH * totalSize));

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < numTilesRows; ++i) {
    for (int j = 0; j < numTilesCols; ++j) {
      vector<vector<int>> Loads(NUM_PES, vector<int>(II_DIST, 0));
      const int curr_tile_offset = tileOffsets[i][j];
      const int curr_tile_size = tileSizes[i][j];
      const CSRMatrix& tile = tiledMatrices[i][j];
      const int num_rows = static_cast<int>(tile.row_offsets.size()) - 1;

      const int num_shared = numSharedRows[i][j];
      vector<vector<int>> sortedShared(num_shared, vector<int>(2, 0));
      for (int k = 0; k < num_shared; ++k) {
        const int row_id = sharedRows[i][j][k];
        sortedShared[k][0] = row_id;
        sortedShared[k][1] = tile.row_offsets[row_id + 1] - tile.row_offsets[row_id];
      }
      std::sort(sortedShared.begin(), sortedShared.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      vector<char> is_shared(num_rows, 0);
      for (int k = 0; k < num_shared; ++k) {
        const int row_id = sharedRows[i][j][k];
        if (0 <= row_id && row_id < num_rows) is_shared[row_id] = 1;
      }

      // 1) Shared rows first (SR=true): distributed across all PEs.
      for (int k = 0; k < num_shared; ++k) {
        const int row_no = sortedShared[k][0];
        const int row_size = sortedShared[k][1];
        const int load_size = CeilDiv(row_size, NUM_PES);
        const uint16_t rowl16 = static_cast<uint16_t>(row_no % NUM_PES);
        const uint16_t rowh16 = static_cast<uint16_t>(row_no / NUM_PES);
        const int row_start = tile.row_offsets[row_no];
        const int row_end = tile.row_offsets[row_no + 1];

        for (int pe = 0; pe < NUM_PES; ++pe) {
          int min_idx = 0;
          int min_val = Loads[pe][0];
          for (int ii = 1; ii < II_DIST; ++ii) {
            if (Loads[pe][ii] < min_val) {
              min_val = Loads[pe][ii];
              min_idx = ii;
            }
          }

          const int ch_no = pe / PES_PER_CH;
          const int inter_ch_pe = pe % PES_PER_CH;
          const uint16_t row16 = (pe & 1) ? rowh16 : rowl16;

          for (int l = 0; l < load_size; ++l) {
            const int ind = row_start + (l * NUM_PES) + pe;
            const int col_id = (ind < row_end) ? tile.col_indices[ind] : 0;
            const float value = (ind < row_end) ? tile.values[ind] : 0.0f;
            const uint32_t val_bits = *reinterpret_cast<const uint32_t*>(&value);

            const int addr = curr_tile_offset + (((Loads[pe][min_idx] * II_DIST) + min_idx) * PES_PER_CH) + inter_ch_pe;
            fpgaAmtx[ch_no][addr] = encode(false, true, true, row16, col_id, val_bits);
            Loads[pe][min_idx]++;
          }
        }
      }

      // 2) Remaining rows (SR=false): packed like prepareAmtx1.
      vector<vector<int>> rem_rows;
      rem_rows.reserve(num_rows);
      for (int r = 0; r < num_rows; ++r) {
        if (is_shared[r]) continue;
        const int row_size = tile.row_offsets[r + 1] - tile.row_offsets[r];
        rem_rows.push_back({r, row_size});
      }
      std::sort(rem_rows.begin(), rem_rows.end(),
                [](const vector<int>& a, const vector<int>& b) { return a[1] > b[1]; });

      for (const auto& rr : rem_rows) {
        const int row_no = rr[0];
        const int pe = row_no % NUM_PES;

        int min_idx = 0;
        int min_val = Loads[pe][0];
        for (int ii = 1; ii < II_DIST; ++ii) {
          if (Loads[pe][ii] < min_val) {
            min_val = Loads[pe][ii];
            min_idx = ii;
          }
        }

        const int ch_no = pe / PES_PER_CH;
        const int inter_ch_pe = pe % PES_PER_CH;
        const uint16_t row16 = static_cast<uint16_t>(row_no / NUM_PES);

        for (int ind = tile.row_offsets[row_no]; ind < tile.row_offsets[row_no + 1]; ++ind) {
          const int col_id = tile.col_indices[ind];
          const float value = tile.values[ind];
          const uint32_t val_bits = *reinterpret_cast<const uint32_t*>(&value);

          const int addr = curr_tile_offset + (((Loads[pe][min_idx] * II_DIST) + min_idx) * PES_PER_CH) + inter_ch_pe;
          fpgaAmtx[ch_no][addr] = encode(false, true, false, row16, col_id, val_bits);
          Loads[pe][min_idx]++;
        }
      }

      // Padding to equalize lane lengths and mark tile end.
      for (int p = 0; p < NUM_PES; ++p) {
        const int ch_no = p / PES_PER_CH;
        const int inter_ch_pe = p % PES_PER_CH;
        for (int ii = 0; ii < II_DIST; ++ii) {
          while (Loads[p][ii] < (curr_tile_size / II_DIST)) {
            const bool tileEnd = (Loads[p][ii] == (curr_tile_size / II_DIST) - 1) && (ii == II_DIST - 1);
            const int col_id = 0;
            const uint16_t row16 = 0;
            const float value = 0.0f;
            const uint32_t val_bits = *reinterpret_cast<const uint32_t*>(&value);

            const int addr = curr_tile_offset + ((Loads[p][ii]++) * II_DIST + ii) * PES_PER_CH + inter_ch_pe;
            fpgaAmtx[ch_no][addr] = encode(tileEnd, false, false, row16, col_id, val_bits);
          }
        }
      }
    }
  }

  return fpgaAmtx;
}

}  // namespace

PrepareAmtxResult PrepareAmtxUnified(const vector<vector<CSRMatrix>>& tiledMatrices,
                                     int numTilesRows,
                                     int numTilesCols,
                                     int tile_depth,
                                     int nnz_total,
                                     const PrepareAmtxConfig& cfg) {
  (void)nnz_total;  // retained for compatibility with legacy signatures / future logging.
  ValidateInputs(tiledMatrices, numTilesRows, numTilesCols, tile_depth);

  PrepareAmtxResult result;

  const bool can_row_share = cfg.kernel_supports_row_sharing;
  const int shared_row_limit = (cfg.shared_row_limit > 0) ? cfg.shared_row_limit : (tile_depth / 2);

  // Always compute no-row-sharing baseline (used for forced-disabled and for Auto comparison).
  const TileWorkStats no_rs = ComputePEloads_NoRowSharing(tiledMatrices, numTilesRows, numTilesCols, tile_depth);
  result.delta_no_row_sharing = no_rs.delta;

  bool use_rs = false;
  TileWorkStats with_rs;
  vector<vector<int>> numSharedRows;
  vector<vector<vector<int>>> sharedRows;

  if (!can_row_share) {
    use_rs = false;
  } else if (cfg.row_sharing == RowSharingPolicy::kForceDisabled) {
    use_rs = false;
  } else {
    // Need shared-rows + with-RS stats (for ForceEnabled and Auto).
    ComputeSharedRowsGreedy(tiledMatrices, numTilesRows, numTilesCols, tile_depth, shared_row_limit,
                            &numSharedRows, &sharedRows);
    with_rs = ComputePEloads_WithRowSharing(tiledMatrices, numTilesRows, numTilesCols, tile_depth,
                                           numSharedRows, sharedRows);
    result.delta_with_row_sharing = with_rs.delta;

    if (cfg.row_sharing == RowSharingPolicy::kForceEnabled) {
      use_rs = true;
    } else {  // Auto
      result.delta_improvement_percent =
          (no_rs.delta > 0.0) ? ((no_rs.delta - with_rs.delta) / no_rs.delta * 100.0) : 0.0;
      use_rs = (result.delta_improvement_percent >= cfg.delta_improvement_threshold_percent);
    }
  }

  result.used_row_sharing = use_rs;

  if (cfg.print_summary) {
    if (!can_row_share) {
      std::cout << "prepareAmtx(unified): row sharing NOT supported by kernel; forcing no-row-sharing path.\n";
    } else if (cfg.row_sharing == RowSharingPolicy::kForceDisabled) {
      std::cout << "prepareAmtx(unified): row sharing forced DISABLED; using no-row-sharing path.\n";
    } else if (cfg.row_sharing == RowSharingPolicy::kForceEnabled) {
      std::cout << "prepareAmtx(unified): row sharing forced ENABLED; using row-sharing path.\n";
    } else {
      std::cout << "prepareAmtx(unified): delta(no-rs)=" << result.delta_no_row_sharing
                << " delta(rs)=" << result.delta_with_row_sharing
                << " improvement=" << result.delta_improvement_percent << "% (threshold="
                << cfg.delta_improvement_threshold_percent << "%) => "
                << (use_rs ? "ENABLE RS\n" : "DISABLE RS\n");
    }
  }

  if (!use_rs) {
    result.fpgaAmtx = PackAmtx_NoRowSharing(tiledMatrices, numTilesRows, numTilesCols, no_rs.tile_sizes);
  } else {
    result.fpgaAmtx = PackAmtx_WithRowSharing(tiledMatrices, numTilesRows, numTilesCols,
                                              numSharedRows, sharedRows, with_rs.tile_sizes);
  }

  if (!result.fpgaAmtx.empty()) {
    result.run_len = static_cast<int>(result.fpgaAmtx[0].size() / PES_PER_CH);
  }

  return result;
}

}  // namespace hispmm_host

// -----------------------------
// Legacy helper_functions.cpp APIs (moved here so `helper_functions.cpp` can be removed)
// -----------------------------

// Function to tile a CSR matrix (ported from legacy helper_functions.cpp).
std::vector<std::vector<CSRMatrix>> tileCSRMatrix(const CSRMatrix& originalMatrix,
                                                  int numRows,
                                                  int numCols,
                                                  int tileRows,
                                                  int tileCols,
                                                  int numTilesRows,
                                                  int numTilesCols) {
  (void)numCols;
  std::vector<std::vector<CSRMatrix>> storedtiledMatrix(numTilesRows, std::vector<CSRMatrix>(numTilesCols));
  for (int i = 0; i < numTilesRows; i++)
    for (int j = 0; j < numTilesCols; j++)
      for (int ii = 0; ii < tileRows + 1; ii++) storedtiledMatrix[i][j].row_offsets.push_back(0);

  for (int row = 0; row < numRows; row++) {
    int tileRow = row / tileRows;
    int tiledRow = row % tileRows;
    for (int j = originalMatrix.row_offsets[row]; j < originalMatrix.row_offsets[row + 1]; j++) {
      int col = originalMatrix.col_indices[j];
      float val = originalMatrix.values[j];
      int tileCol = col / tileCols;
      int tiledCol = col % tileCols;

      storedtiledMatrix[tileRow][tileCol].col_indices.push_back(tiledCol);
      storedtiledMatrix[tileRow][tileCol].values.push_back(val);
      storedtiledMatrix[tileRow][tileCol].row_offsets[tiledRow + 1]++;
    }
  }

  for (int i = 0; i < numTilesRows; i++)
    for (int j = 0; j < numTilesCols; j++)
      for (int ii = 1; ii < tileRows + 1; ii++)
        storedtiledMatrix[i][j].row_offsets[ii] += storedtiledMatrix[i][j].row_offsets[ii - 1];

  return storedtiledMatrix;
}

// function from Serpens and functions to read mtx file (ported from legacy helper_functions.cpp)
static int cmp_by_column_row(const void* aa, const void* bb) {
  rcv* a = (rcv*)aa;
  rcv* b = (rcv*)bb;

  if (a->c > b->c) return +1;
  if (a->c < b->c) return -1;

  if (a->r > b->r) return +1;
  if (a->r < b->r) return -1;

  return 0;
}

static void sort_by_fn(int nnz_s,
                       std::vector<int>& cooRowIndex,
                       std::vector<int>& cooColIndex,
                       std::vector<float>& cooVal,
                       int (*cmp_func)(const void*, const void*)) {
  rcv* rcv_arr = new rcv[nnz_s];

  for (int i = 0; i < nnz_s; ++i) {
    rcv_arr[i].r = cooRowIndex[i];
    rcv_arr[i].c = cooColIndex[i];
    rcv_arr[i].v = cooVal[i];
  }

  qsort(rcv_arr, nnz_s, sizeof(rcv), cmp_func);

  for (int i = 0; i < nnz_s; ++i) {
    cooRowIndex[i] = rcv_arr[i].r;
    cooColIndex[i] = rcv_arr[i].c;
    cooVal[i] = rcv_arr[i].v;
  }

  delete[] rcv_arr;
}

static void mm_init_read(FILE* f, char* filename, MM_typecode& matcode, int& m, int& n, int& nnz) {
  if (mm_read_banner(f, &matcode) != 0) {
    std::cout << "Could not process Matrix Market banner for " << filename << std::endl;
    exit(1);
  }

  int ret_code;
  if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
    std::cout << "Could not read Matrix Market format for " << filename << std::endl;
    exit(1);
  }
}

static void load_S_matrix(FILE* f_A,
                          int nnz_mmio,
                          int& nnz,
                          std::vector<int>& cooRowIndex,
                          std::vector<int>& cooColIndex,
                          std::vector<float>& cooVal,
                          MM_typecode& matcode) {
  if (mm_is_complex(matcode)) {
    std::cout << "Reading in a complex matrix, not supported yet!" << std::endl;
    exit(1);
  }

  if (!mm_is_symmetric(matcode)) {
    std::cout << "It's an NS matrix.\n";
  } else {
    std::cout << "It's an S matrix.\n";
  }

  int r_idx, c_idx;
  float value;
  int idx = 0;

  for (int i = 0; i < nnz_mmio; ++i) {
    if (mm_is_pattern(matcode)) {
      fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
      value = 1.0;
    } else {
      fscanf(f_A, "%d %d %f\n", &r_idx, &c_idx, &value);
    }

    unsigned int* tmpPointer_v = reinterpret_cast<unsigned int*>(&value);
    unsigned int uint_v = *tmpPointer_v;
    if (uint_v != 0) {
      if (r_idx < 1 || c_idx < 1) {
        std::cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << std::endl;
        exit(1);
      }

      cooRowIndex[idx] = r_idx - 1;
      cooColIndex[idx] = c_idx - 1;
      cooVal[idx] = value;
      idx++;

      if (mm_is_symmetric(matcode)) {
        if (r_idx != c_idx) {
          cooRowIndex[idx] = c_idx - 1;
          cooColIndex[idx] = r_idx - 1;
          cooVal[idx] = value;
          idx++;
        }
      }
    }
  }
  nnz = idx;
}

void readMatrixCSC(char* filename,
                   std::vector<float>& values,
                   std::vector<int>& rowIndices,
                   std::vector<int>& colOffsets,
                   int& rows,
                   int& cols,
                   int& nnz) {
  int nnz_mmio;
  MM_typecode matcode;
  FILE* f_A;

  if ((f_A = fopen(filename, "r")) == NULL) {
    std::cout << "Could not open " << filename << std::endl;
    exit(1);
  }

  mm_init_read(f_A, filename, matcode, rows, cols, nnz_mmio);

  if (!mm_is_coordinate(matcode)) {
    std::cout << "The input matrix file " << filename << "is not a coordinate file!" << std::endl;
    exit(1);
  }

  int nnz_alloc = (mm_is_symmetric(matcode)) ? (nnz_mmio * 2) : nnz_mmio;
  std::vector<int> cooRowIndex(nnz_alloc);
  std::vector<int> cooColIndex(nnz_alloc);
  values.resize(nnz_alloc);

  load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, values, matcode);
  fclose(f_A);

  sort_by_fn(nnz, cooRowIndex, cooColIndex, values, cmp_by_column_row);

  // convert to CSC format
  int M_K = cols;
  colOffsets.resize(M_K + 1);
  std::vector<int> counter(M_K, 0);

  for (int i = 0; i < nnz; i++) {
    counter[cooColIndex[i]]++;
  }

  colOffsets[0] = 0;
  for (int i = 1; i <= M_K; i++) {
    colOffsets[i] = colOffsets[i - 1] + counter[i - 1];
  }

  rowIndices.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    rowIndices[i] = cooRowIndex[i];
  }

  if (mm_is_symmetric(matcode)) {
    values.resize(nnz);
  }
}

void convertCSCtoCSR(const std::vector<float>& cscValues,
                     const std::vector<int>& cscRowIndices,
                     const std::vector<int>& cscColOffsets,
                     std::vector<float>& csrValues,
                     std::vector<int>& csrColIndices,
                     std::vector<int>& csrRowOffsets,
                     int rows,
                     int cols,
                     int nnz) {
  (void)cscValues;
  csrValues.resize(nnz);
  csrColIndices.resize(nnz);
  csrRowOffsets.resize(rows + 1);
  std::vector<int> rowCounts(rows, 0);

  for (int i = 0; i < nnz; i++) {
    rowCounts[cscRowIndices[i]]++;
  }

  csrRowOffsets[0] = 0;
  for (int i = 0; i < rows; i++) {
    csrRowOffsets[i + 1] = csrRowOffsets[i] + rowCounts[i];
  }

  std::vector<int> rowOffset(rows, 0);
  for (int j = 0; j < cols; j++) {
    for (int i = cscColOffsets[j]; i < cscColOffsets[j + 1]; i++) {
      int row = cscRowIndices[i];
      int index = csrRowOffsets[row] + rowOffset[row];
      csrValues[index] = 1.0;
      csrColIndices[index] = j;
      rowOffset[row]++;
    }
  }
}

void printMatrixCSR(std::vector<float> values,
                    std::vector<int> columns,
                    std::vector<int> rowPtr,
                    int numRows,
                    int numCols) {
  std::cout << "Matrix in dense format:" << std::endl;
  for (int i = 0; i < numRows; i++) {
    int prev_col = 0;
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      int col = columns[j];
      float val = values[j];
      for (int k = prev_col; k < col; k++) printf("%.4f; ", 0.0);
      printf("%.4f; ", val);
      prev_col = col + 1;
    }
    for (int k = prev_col; k < numCols; k++) printf("%.4f; ", 0.0);
    printf("\n");
  }
}

// Legacy `prepareAmtx` wrapper: keep signature, route to unified implementation.
std::vector<aligned_vector<uint64_t>> prepareAmtx(std::vector<std::vector<CSRMatrix>> tiledMatrices,
                                                  const int numTilesRows,
                                                  const int numTilesCols,
                                                  const int Depth,
                                                  const int Window,
                                                  const int rows,
                                                  const int cols,
                                                  const int nnz) {
  (void)Window;
  (void)rows;
  (void)cols;

  hispmm_host::PrepareAmtxConfig cfg;
  // Generic wrapper: keep quiet and conservative by default.
  // Codegen should prefer calling `PrepareAmtxUnified(...)` directly and setting cfg explicitly.
  cfg.kernel_supports_row_sharing = false;
  cfg.row_sharing = hispmm_host::RowSharingPolicy::kForceDisabled;
  cfg.print_summary = false;

  const auto res = hispmm_host::PrepareAmtxUnified(tiledMatrices, numTilesRows, numTilesCols, Depth, nnz, cfg);
  return res.fpgaAmtx;
}


