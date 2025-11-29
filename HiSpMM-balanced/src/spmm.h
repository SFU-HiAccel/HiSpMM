#ifndef SPMV_H
#define SPMV_H
#include <cstdint>
#include <tapa.h>
#include <ap_int.h>
#include <numeric>
#include <math.h>
// #define PE64_WOHRD
#define NUM_A_CH 10 //6
#define NUM_C_CH 4 //4
#define NUM_B_CH 4 //4

#define LOG_2_NUM_PES 7
#define NUM_PES 80 //48
#define NUM_PES_HALF (NUM_PES/2)
#define GROUP_SIZE 4

#define NUM_PEG (NUM_PES/GROUP_SIZE)
#define PADDING 1
#define II_DIST 8
#define LAT (II_DIST-2)
#define FIFO_DEPTH 2
#define FIFO_LARGE_DEPTH 11
#define PES_PER_CH 8


#define N0 8
#define B_CHUNK_SIZE (2*N0) //2*N0
#define K0 ((NUM_B_CH*8*1024)/N0) //4096
#define KX ((NUM_B_CH*B_CHUNK_SIZE)/N0) //8
#define B_READ_LEN (((K0*N0)/NUM_B_CH)/B_CHUNK_SIZE) //512
#define U 2 //#URAMS = NUM_PES * U * (N0/2) = 64 * 3 * 4 = 768/960
#define MAX_ROWS_PER_PE (U*4096)
#define M0 (NUM_PES * MAX_ROWS_PER_PE)
#define C_READ_LEN ((M0*N0)/NUM_C_CH/B_CHUNK_SIZE)
#define MX (NUM_C_CH*16/N0)

#define TEMP_BUFFER_SIZE 8

using uint64_v = tapa::vec_t<uint64_t, PES_PER_CH>;
using uint64_v2 = tapa::vec_t<uint64_t, 2>;
// using float_v8 = tapa::vec_t<float, N0>;
using float_vN = tapa::vec_t<float, N0>;
// using uint16_v2 = tapa::vec_t<uint16_t, 2>;
// using float_v16 = tapa::vec_t<float, 16>;


// 3) Now define a custom vector type for B. 
// using float_v16 = tapa::vec_t<float, B_CHUNK_SIZE>;
using float_vB = tapa::vec_t<float, B_CHUNK_SIZE>;

struct flags_pkt {
    bool sharedRow;
    bool tileEnd;
    bool last;
};

struct Cnoc_pkt {
    bool dummy;
    bool last;
    bool tileEnd;
    bool sharedRow;
    uint16_t row16;
    uint8_t bank;
    float val[8];
};


struct Cvec_pkt {
    bool dummy;
    bool tileEnd;
    uint16_t row16;
    float val[8];
};

void hispmm(tapa::mmaps<uint64_v, NUM_A_CH> A,
          tapa::mmaps<float_vB, NUM_B_CH> B,
          tapa::mmaps<float_vB, NUM_C_CH> c_in,
          tapa::mmaps<float_vB, NUM_C_CH> c_out,
          const float alpha, const float beta,
          const uint32_t M, const uint32_t N, const uint32_t K,
          const uint32_t numTilesM, const uint32_t numTilesN, const uint32_t numTilesK,
          const uint32_t len, const uint32_t last_tile_idx, const uint16_t rp_time);
#endif
