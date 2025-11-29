#include "helper_functions.h"
#include "spmm.h"
#include "mmio.h"
#include <cmath>


std::vector<std::vector<int>> computePEloads1(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, int& totalSize)
{
    std::vector<std::vector<int>> tileSizes(numTilesRows, std::vector<int>(numTilesCols, 0));

    // Accumulators for PE-load statistics across all tiles
    double load_sum = 0.0;
    double load_sq_sum = 0.0;
    long long load_count = 0;

    #pragma omp parallel for collapse(2) reduction(+:load_sum, load_sq_sum, load_count)
    for(int i = 0; i < numTilesRows; i++) {
        for(int j = 0; j < numTilesCols; j++) {
            CSRMatrix* curr_tile = &tiledMatrices[i][j];
            int num_rows = curr_tile->row_offsets.size() - 1;
            std::vector<std::vector<int>> Loads(NUM_PES, std::vector<int>(II_DIST, 0));
            std::vector<std::vector<int>> sorted_rows(num_rows, std::vector<int>(2, 0));

            for(int row = 0; row < num_rows; row++) {
                sorted_rows[row][0] = row;
                sorted_rows[row][1] = curr_tile->row_offsets[row+1] - curr_tile->row_offsets[row];
            }

            sort(sorted_rows.begin(), 
                sorted_rows.end(),
                [] (const std::vector<int> &a, const std::vector<int> &b)
                {
                    return a[1] > b[1];
                });


            for(int row = 0; row < num_rows; row++) {
                int rowSize = sorted_rows[row][1];
                int pe = sorted_rows[row][0] % NUM_PES;
                int min = Loads[pe][0];
                int min_idx = 0;
                for(int ii = 0; ii < II_DIST; ii++) {
                    if(Loads[pe][ii] < min){
                        min = Loads[pe][ii];
                        min_idx = ii;
                    }
                }
                Loads[pe][min_idx] += rowSize;
            }

            // Accumulate PE loads for statistics (after Loads is fully built)
            for (int p = 0; p < NUM_PES; ++p) {
                for (int ii = 0; ii < II_DIST; ++ii) {
                    double load = static_cast<double>(Loads[p][ii]);
                    load_sum    += load;
                    load_sq_sum += load * load;
                    load_count  += 1;
                }
            }

            int max_load = Loads[0][0];

            for(int p = 0; p < NUM_PES; p++) 
                for(int ii = 0; ii < II_DIST; ii++) 
                    if(Loads[p][ii] > max_load) 
                        max_load = Loads[p][ii];

            tileSizes[i][j] = (max_load+ PADDING) * II_DIST;
        }
    }

    for(int i = 0; i < numTilesRows; i++) 
        for(int j = 0; j < numTilesCols; j++) 
            totalSize += tileSizes[i][j];

    // Compute mean, stddev, and delta (stddev / mean) for PE loads
    if (load_count > 0) {
        double mean = load_sum / static_cast<double>(load_count);
        double var  = load_sq_sum / static_cast<double>(load_count) - mean * mean;
        if (var < 0.0) var = 0.0;  // numerical safety
        double stddev = std::sqrt(var);
        double delta  = (mean != 0.0) ? (stddev / mean) : 0.0;

        std::cout << "PE load statistics: mean = " << ceil(mean)
                  << ", stddev = " << stddev
                  << ", delta = " << delta << std::endl;
    }

    return tileSizes;
}


std::vector<aligned_vector<uint64_t>> prepareAmtx1(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    std::vector<std::vector<int>> tileSizes)
{
  std::vector<std::vector<int>> tileOffsets (numTilesRows, std::vector<int>(numTilesCols, 0));
  
  int totalSize = 0;
  for(int i = 0; i < numTilesRows; i++) {
    for(int j = 0; j < numTilesCols; j++) {
      tileOffsets[i][j] = PES_PER_CH * totalSize;
      totalSize += tileSizes[i][j];
    }
  }

  std::vector<aligned_vector<uint64_t>> fpgaAmtx(NUM_A_CH, aligned_vector<uint64_t>(PES_PER_CH * totalSize));

  #pragma omp parallel for num_threads(48) collapse(2)
  for(int i = 0; i < numTilesRows; i++) {
    for(int j = 0; j < numTilesCols; j++) {

      std::vector<std::vector<int>> Loads(NUM_PES, std::vector<int>(II_DIST, 0));
      // printf("Tile[%d][%d] Size: %d\n", i, j, tileSizes[i][j]);
      int curr_tile_offset = tileOffsets[i][j];
      int curr_tile_size = tileSizes[i][j];
      CSRMatrix* curr_tile = &tiledMatrices[i][j];
      int num_rows = curr_tile->row_offsets.size() - 1;
      std::vector<std::vector<int>> sorted_rows(num_rows, std::vector<int>(2, 0));

      for(int row = 0; row < num_rows; row++) {
        sorted_rows[row][0] = row;
        sorted_rows[row][1] = curr_tile->row_offsets[row+1] - curr_tile->row_offsets[row];
      }

      sort(sorted_rows.begin(), 
          sorted_rows.end(),
          [] (const std::vector<int> &a, const std::vector<int> &b)
          {
              return a[1] > b[1];
          });


      // data phase
      for(int row = 0; row < num_rows; row++) {
        int rowSize = sorted_rows[row][1];
        int row_no = sorted_rows[row][0];
        int pe = row_no % NUM_PES;
        
        int min = Loads[pe][0];
        int min_idx = 0;
        for(int ii = 0; ii < II_DIST; ii++) {
          if(Loads[pe][ii] < min){
            min = Loads[pe][ii];
            min_idx = ii;
          }
        }
        
        int ch_no = pe / PES_PER_CH;
        int inter_ch_pe = pe % PES_PER_CH;
        uint16_t row16 = (row_no / NUM_PES);
        for(int ind = curr_tile->row_offsets[row_no]; ind < curr_tile->row_offsets[row_no+1]; ind++) {
          int col_id = curr_tile->col_indices[ind];
          float value = curr_tile->values[ind];
          uint32_t val_bits = *(uint32_t*)&value;
          int addr = curr_tile_offset + (((Loads[pe][min_idx] * II_DIST) + min_idx) * PES_PER_CH) + inter_ch_pe;
          fpgaAmtx[ch_no][addr] = encode(false, true, false, row16, col_id, val_bits);
          Loads[pe][min_idx]++;
        }  
      }

// padding phase
      for(int p = 0; p < NUM_PES; p++) {
        int ch_no = p / PES_PER_CH;
        int inter_ch_pe = p % PES_PER_CH;
        for(int ii = 0; ii < II_DIST; ii++) {
          while(Loads[p][ii] < (curr_tile_size/II_DIST)) {
            bool tileEnd = (Loads[p][ii] == (curr_tile_size/II_DIST) - 1) && (ii == II_DIST-1);
            int col_id = 0;
            uint16_t row16 = 0;
            float value = 0;
            uint32_t val_bits = *(uint32_t*)&value;
            int addr = curr_tile_offset + ((Loads[p][ii]++) * II_DIST + ii) * PES_PER_CH + inter_ch_pe;
            fpgaAmtx[ch_no][addr] = encode(tileEnd, false, false, row16, col_id, val_bits);
          }
        }
      }
    }
  }

  return fpgaAmtx;
}


std::vector<aligned_vector<uint64_t>> prepareAmtx(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    const int Depth, const int Window, const int rows, const int cols, const int nnz) 
{    
    int run_len1 = 0;
    std::vector<std::vector<int>> tileSizes1 = computePEloads1(tiledMatrices, numTilesRows, numTilesCols, run_len1);
    printf("NUM_PES is 80, continuing without dense row sharing...\n");
    return prepareAmtx1(tiledMatrices, numTilesRows, numTilesCols, tileSizes1);

    
}

// Function to tile a CSR matrix
std::vector<std::vector<CSRMatrix>> tileCSRMatrix(const CSRMatrix& originalMatrix, int numRows, int numCols, int tileRows, int tileCols, int numTilesRows, int numTilesCols) {
    std::vector<std::vector<CSRMatrix>> tiledMatrices;

    std::vector<std::vector<CSRMatrix>> storedtiledMatrix(numTilesRows, std::vector<CSRMatrix>(numTilesCols));
    for (int i = 0; i < numTilesRows; i++) 
        for (int j = 0; j < numTilesCols; j++) 
            for(int ii = 0; ii < tileRows + 1; ii++) 
                storedtiledMatrix[i][j].row_offsets.push_back(0);

    for (int row = 0; row < numRows; row++)
    {
        int tileRow = row / tileRows;
        int tiledRow = row % tileRows;
        for (int j = originalMatrix.row_offsets[row]; j < originalMatrix.row_offsets[row+1]; j++)
        {
            int col = originalMatrix.col_indices[j];
            float val = originalMatrix.values[j];
            int tileCol = col / tileCols;
            int tiledCol = col % tileCols;

            storedtiledMatrix[tileRow][tileCol].col_indices.push_back(tiledCol);
            storedtiledMatrix[tileRow][tileCol].values.push_back(val);
            storedtiledMatrix[tileRow][tileCol].row_offsets[tiledRow+1]++;
        }
    }

    for (int i = 0; i < numTilesRows; i++) 
        for (int j = 0; j < numTilesCols; j++) 
            for(int ii = 1; ii < tileRows + 1; ii++) 
                storedtiledMatrix[i][j].row_offsets[ii] += storedtiledMatrix[i][j].row_offsets[ii-1];

    return storedtiledMatrix;
}

// function from Serpens and functions to read mtx file
int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv * a = (rcv *) aa;
    rcv * b = (rcv *) bb;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    return 0;
}

void sort_by_fn(int nnz_s,
                std::vector<int> & cooRowIndex,
                std::vector<int> & cooColIndex,
                std::vector<float> & cooVal,
                int (* cmp_func)(const void *, const void *)) {
    rcv * rcv_arr = new rcv[nnz_s];
    
    for(int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }
    
    qsort(rcv_arr, nnz_s, sizeof(rcv), cmp_func);
    
    for(int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }
    
    delete [] rcv_arr;
}

void mm_init_read(FILE * f,
                  char * filename,
                  MM_typecode & matcode,
                  int & m,
                  int & n,
                  int & nnz) {

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

void load_S_matrix(FILE* f_A,
                   int nnz_mmio,
                   int & nnz,
                   std::vector<int> & cooRowIndex,
                   std::vector<int> & cooColIndex,
                   std::vector<float> & cooVal,
                   MM_typecode & matcode) {
    
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
        }else {
            fscanf(f_A, "%d %d %f\n", &r_idx, &c_idx, &value);
        }
        
        unsigned int * tmpPointer_v = reinterpret_cast<unsigned int*>(&value);;
        unsigned int uint_v = *tmpPointer_v;
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) { // report error
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

void readMatrixCSC(char* filename, std::vector<float>& values, std::vector<int>& rowIndices, std::vector<int>& colOffsets, int& rows, int& cols, int& nnz) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE * f_A;
    
    if ((f_A = fopen(filename, "r")) == NULL) {
        std::cout << "Could not open " << filename << std::endl;
        exit(1);
    }
    
    mm_init_read(f_A, filename, matcode, rows, cols, nnz_mmio);
    
    if (!mm_is_coordinate(matcode)) {
        std::cout << "The input matrix file " << filename << "is not a coordinate file!" << std::endl;
        exit(1);
    }
    
    int nnz_alloc = (mm_is_symmetric(matcode))? (nnz_mmio * 2): nnz_mmio;
    //std::cout << "Matrix A -- #row: " << rows << " #col: " << cols << std::endl;
    
    std::vector<int> cooRowIndex(nnz_alloc);
    std::vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    values.resize(nnz_alloc);
    
    //std::cout << "Loading input matrix A from " << filename << "\n";
    
    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, values, matcode);
    
    fclose(f_A);
    
    sort_by_fn(nnz, cooRowIndex, cooColIndex, values, cmp_by_column_row);
    
    // convert to CSC format
    int M_K = cols;
    colOffsets.resize(M_K+1);
    std::vector<int> counter(M_K, 0);
    
    for (int i = 0; i < nnz; i++) {
        counter[cooColIndex[i]]++;
    }
    
    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
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
        //eleIndex.resize(nnz);
        values.resize(nnz);
    }
}

void convertCSCtoCSR(const std::vector<float>& cscValues, const std::vector<int>& cscRowIndices, const std::vector<int>& cscColOffsets,
                     std::vector<float>& csrValues, std::vector<int>& csrColIndices, std::vector<int>& csrRowOffsets, int rows, int cols, int nnz) {
    // allocate memory
    csrValues.resize(nnz);
    csrColIndices.resize(nnz);
    csrRowOffsets.resize(rows + 1);
    std::vector<int> rowCounts(rows, 0);

    for (int i = 0; i < nnz; i++) {
        rowCounts[cscRowIndices[i]]++;
    }

    // convert rowCounts to cumulative sum
    csrRowOffsets[0] = 0;
    for (int i = 0; i < rows; i++) {
      csrRowOffsets[i+1] = csrRowOffsets[i] + rowCounts[i];
    }

    std::vector<int> rowOffset(rows, 0);
    // fill csrValues and csrColIndices
    for (int j = 0; j < cols; j++) {
        for (int i = cscColOffsets[j]; i < cscColOffsets[j + 1]; i++) {
            int row = cscRowIndices[i];
            int index = csrRowOffsets[row] + rowOffset[row];
            csrValues[index] = /*cscValues[i]*/ 1.0;
            csrColIndices[index] = j;
            rowOffset[row]++;
        }
    }
}

void printMatrixCSR(std::vector<float> values, std::vector<int> columns, std::vector<int> rowPtr, int numRows, int numCols) {
    // Print the matrix in CSR format
    std::cout << "Matrix in dense format:" << std::endl;
    for (int i = 0; i < numRows; i++) {
        int prev_col = 0;
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
            int col = columns[j];
            float val = values[j];
            for (int k = prev_col; k < col; k++)
                printf("%.4f; ", 0.0);
            printf("%.4f; ",val);
            prev_col = col + 1;
        }
        for (int k = prev_col; k < numCols; k++)
                printf("%.4f; ", 0.0);
        printf("\n");
    }
}

