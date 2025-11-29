#include "spmm.h" 

inline void async_readA(tapa::async_mmap<uint64_v> & A,
                       tapa::ostreams<uint64_v2, PES_PER_CH/2> & fifo_A,
                       const uint32_t A_len,
                       uint32_t& i_req,
                       uint32_t& i_resp) {
    #pragma HLS inline
    uint64_v tmp;
    if ((i_req < A_len) &
        !A.read_addr.full()) {
        A.read_addr.try_write(i_req);
        ++i_req;
    }

    bool full = 0;
    for(int a = 0; a < PES_PER_CH/2; a++)
    #pragma HLS UNROLL
        full |= fifo_A[a].full();

    if (!full & !A.read_data.empty()) {
        A.read_data.try_read(tmp);
        for(int a = 0; a < PES_PER_CH/2; a++) {
        #pragma HLS UNROLL
            uint64_v2 t;
            t[0] = tmp[a*2];
            t[1] = tmp[a*2 + 1];
            fifo_A[a].try_write(t);
        }
        ++i_resp;
    }
}

inline void async_readB(tapa::async_mmap<float_vB> & B,
                       tapa::ostream<float_vB> & fifo_B,
                       const uint32_t B_len,
                       uint32_t & i_req,
                       uint32_t & i_resp) {
    #pragma HLS inline
    if ((i_req < B_len) &
        !B.read_addr.full()) {
        B.read_addr.try_write(i_req);
        ++i_req;
    }

    if (!fifo_B.full() & !B.read_data.empty()) {
        float_vB tmp;
        B.read_data.try_read(tmp);
        fifo_B.try_write(tmp);
        ++i_resp;
    }
}

inline void async_readC(tapa::async_mmap<float_vB> & C,
                       tapa::ostream<float_vB> & fifo_C,
                       const uint32_t C_len,
                       uint32_t & i_req,
                       uint32_t & i_resp) {
    #pragma HLS inline
    if ((i_req < C_len) &
        !C.read_addr.full()) {
        C.read_addr.try_write(i_req);
        ++i_req;
    }

    if (!fifo_C.full() & !C.read_data.empty()) {
        float_vB tmp;
        C.read_data.try_read(tmp);
        fifo_C.try_write(tmp);
        ++i_resp;
    }
}

inline void async_writeC(tapa::async_mmap<float_vB>& mem,
    tapa::istream<float_vB>& fifo,
    uint32_t start,
    uint32_t count) {
    #pragma HLS inline

  for(int i_req = start, i_resp = 0; i_resp < count;) {
    #pragma HLS pipeline II=1
    // issue write requests
    float_vB tmp;

    if (i_req < (start + count) &&
        !fifo.empty() &&
        !mem.write_addr.full() &&
        !mem.write_data.full()) {
    
   
      tmp = fifo.read(nullptr);

      mem.write_addr.try_write(i_req);
      mem.write_data.try_write(tmp);
      ++i_req;
    }

    // receive acks of write success
    if (!mem.write_resp.empty()) {
      i_resp += unsigned(mem.write_resp.read(nullptr)) + 1;
    }
  }
} 

void MM2S_A(tapa::async_mmap<uint64_v>& mmap,
    tapa::ostreams<uint64_v2, PES_PER_CH/2>& streams,
    const uint32_t len, const uint16_t numTilesN, const uint16_t rp_time) {
    printf("MM2S_A\n");
    for(uint32_t rp = 0; rp < rp_time * numTilesN; rp++) {
        for(uint32_t i_req = 0, i_resp = 0; i_resp < len;) {
        #pragma HLS pipeline II=1
        async_readA(mmap,
                    streams,
                    len,
                    i_req, i_resp); 
         
        }
    }
     
    printf("MM2S_A done\n");
}

void MM2S_B(tapa::async_mmap<float_vB>& mmap,
    tapa::ostream<float_vB>& stream,
    const uint32_t numTilesM, const uint32_t numTilesN, const uint32_t numTilesK,
    const uint32_t K, const uint16_t rp_time) {
  for (int r = 0; r < rp_time; r++) {
    for (int n = 0; n < numTilesN; n++) {
      for (int m = 0; m < numTilesM; m++) {
        for (int k = 0; k < numTilesK; k++) {
          #pragma HLS loop_flatten OFF
          int read_len = ((k == (numTilesK - 1)) && (K%K0 != 0)) ? ((K%K0)/KX) : B_READ_LEN; //last tile might not be having B_REAd_LEN adresses
          int start_addr = (k * numTilesN * B_READ_LEN) + (n * read_len); // each tile in a channel takes B_READ_LEN addresses except last k tiles
          int end_addr = start_addr + read_len;
          for(uint32_t i_req = start_addr, i_resp = 0; i_resp < read_len;) {
            #pragma HLS pipeline II=1
            async_readB(mmap,
                        stream,
                        end_addr,
                        i_req, i_resp); 
          }
        }
      }
    } 
  }
  printf("MM2S_B done\n");
}

void MM2S_C(tapa::async_mmap<float_vB>& mmap,
    tapa::ostream<float_vB>& stream,
    const uint32_t numTilesM, const uint32_t numTilesN, 
    const uint32_t M, const uint32_t N, const uint16_t rp_time) {
  for (int r = 0; r < rp_time; r++) {
    for (int n = 0; n < numTilesN; n++) {
      for (int m = 0; m < numTilesM; m++) {    
        #pragma HLS loop_flatten OFF          
        int read_len = ((m == (numTilesM - 1)) && (M%M0 != 0)) ? ((M%M0)/MX) : C_READ_LEN; //last tile might not be having C_REAd_LEN adresses
        int start_addr = (m * numTilesN * C_READ_LEN) + (n * read_len); // each tile in a channel takes C_READ_LEN addresses except last k tiles
        int end_addr = start_addr + read_len;
        for(uint32_t i_req = start_addr, i_resp = 0; i_resp < read_len;) {
          #pragma HLS pipeline II=1
          async_readB(mmap,
            stream,
            end_addr,
            i_req, i_resp); 
        }
      }
    }
  }
  printf("MM2S_C done\n");
}

void S2MM_C(tapa::istream<float_vB>& stream,
    tapa::async_mmap<float_vB>& mmap,
    const uint32_t numTilesM, const uint32_t numTilesN, 
    const uint32_t M, const uint32_t N, const uint16_t rp_time) {
    for (int r = 0; r < rp_time; r++) {
      for (int n = 0; n < numTilesN; n++) {
        for (int m = 0; m < numTilesM; m++) {     
          #pragma HLS loop_flatten OFF         
          int read_len = ((m == (numTilesM - 1)) && (M%M0 != 0)) ? ((M%M0)/MX) : C_READ_LEN; //last tile might not be having C_REAd_LEN adresses
          int start_addr = (m * numTilesN * C_READ_LEN) + (n * read_len); // each tile in a channel takes C_READ_LEN addresses except last k tiles
          async_writeC(mmap, stream, start_addr, read_len);
        }
      }
    }
    printf("S2MM_C done\n");
}
    

void DummyRead(tapa::istream<float_vB>& b_in) {
  for (;;) {
  #pragma HLS PIPELINE II=1
    float_vB temp = b_in.read();
  }
}

void Accumulator(
  tapa::ostream<float_vN>& c_out,
  tapa::istream<Cvec_pkt>& c_in,
  const uint32_t M, const uint32_t numTilesM, const uint32_t last_tile_idx) {

  float_vN buffer_C[MAX_ROWS_PER_PE];
  #pragma HLS bind_storage variable=buffer_C type=RAM_2P impl=URAM

  uint32_t tm = 0;
  bool lastTileM = false;
  for(int i = 0; i <= last_tile_idx; i++) {
    #pragma HLS PIPELINE OFF

    lastTileM = (tm == (numTilesM - 1));
    int read_len = ((lastTileM) && (M%M0 != 0)) ? ((M%M0)/NUM_PES) : MAX_ROWS_PER_PE; //last tile might not be having C_REAd_LEN adresses 
    //init
    init_c:for(int m = 0; m < read_len; m++) 
      #pragma HLS PIPELINE II=1
      for(int n = 0; n < N0; n++) 
        buffer_C[m][n] = 0;
      
    //compute
    acc_c:for(bool tileEnd = false; !(tileEnd); ) {
      #pragma HLS PIPELINE II=1
      #pragma HLS dependence variable=buffer_C inter RAW distance=8 true
      Cvec_pkt temp_in = c_in.read();

      tileEnd = temp_in.tileEnd;

      bool dummy = temp_in.dummy;
      int m = temp_in.row16;
      if (m > M0)
        printf("m = %d\n", m);

      if (!dummy) 
      {
        for(int n = 0; n < N0; n++) {
          buffer_C[m][n] += temp_in.val[n];
        }
      }
    }
    
    //update
    
    updt_C:for(int l = 0; l < read_len; l++) {
    #pragma HLS PIPELINE II=1
      c_out << buffer_C[l];
    }
    tm = (lastTileM) ? 0 : tm + 1;
  } //i
}

void PEG(tapa::istreams<uint64_v2, GROUP_SIZE/2>& a_in, 
  tapa::istreams<float_vB, NUM_B_CH>& b_in,
  tapa::ostreams<float_vB, NUM_B_CH>& b_out,
  tapa::ostreams<Cvec_pkt, GROUP_SIZE>& c_out,
  const uint32_t K, const uint32_t numTilesK, const uint32_t last_tile_idx) {


  float buff_B[GROUP_SIZE/2][N0][B_CHUNK_SIZE / N0][NUM_B_CH * B_READ_LEN];
  #pragma HLS bind_storage variable=buff_B type=RAM_T2P impl=BRAM
  #pragma HLS array_partition variable=buff_B type=complete dim=1
  #pragma HLS array_partition variable=buff_B type=complete dim=2
  #pragma HLS array_partition variable=buff_B type=complete dim=3
  #pragma HLS array_partition variable=buff_B type=cyclic factor=2 dim=4
  bool last = false; 
  bool lastTikeK = false;
  uint32_t tK = 0;

  for (uint32_t i = 0; !last;) {
    #pragma HLS PIPELINE OFF
    uint64_v2 temp_in[GROUP_SIZE/2];
    float val_in[GROUP_SIZE];
    ap_uint<13> col_id[GROUP_SIZE];
    float_vN temp_out[GROUP_SIZE];
    uint16_t row_out[GROUP_SIZE];
    flags_pkt temp_flag;
    lastTikeK = (tK == (numTilesK - 1));
    int read_len = (lastTikeK && (K%K0 != 0)) ? (K%K0/KX) : B_READ_LEN; 
    load_B:for (int l = 0; l < read_len; l++) {
    #pragma HLS PIPELINE II=1
      for (int ch = 0; ch < NUM_B_CH; ch++) {
        float_vB temp = b_in[ch].read();
        for (int p = 0; p < B_CHUNK_SIZE; p++) {
          for(int g = 0; g < GROUP_SIZE/2; g++){
            buff_B[g][p%N0][p/N0][l*NUM_B_CH + ch] = temp[p];
          }
        }
        b_out[ch].write(temp);
      }
    }

    printf("Load B done: %d / %d\n", i, last_tile_idx);
    mul_AB:for (bool tileEnd = false; !(tileEnd); ) {
    #pragma HLS loop_tripcount min=1 max=100000
    #pragma HLS PIPELINE II=1
      for(int g = 0; g < GROUP_SIZE/2; g++)
        temp_in[g] = a_in[g].read();

      
      tileEnd = (temp_in[0][0] >> 47) & 1;
      last = (i==last_tile_idx) & tileEnd & lastTikeK;
      temp_flag.tileEnd = tileEnd & lastTikeK;
      
      for(int p = 0; p < GROUP_SIZE; p++) {
      #pragma HLS UNROLL
          uint64_t a = temp_in[p/2][p%2];
          uint32_t val_bits = a & 0xFFFFFFFF;
          val_in[p] = *(float*)(&val_bits);
          col_id[p] = (a >> 32) & 0x1FFF;
          uint16_t row = (a >> 48) & 0xFFFF;
          bool rowEnd  = (a >> 46) & 1;
          row_out[p] = row | ((uint16_t)rowEnd << 15);
          
      }

      for(int p = 0; p < GROUP_SIZE; p++) {
        Cvec_pkt temp;
      #pragma HLS UNROLL
        for (int n = 0; n < N0; n++) {
          temp.val[n] = val_in[p] * buff_B[p/2][n][col_id[p]%2][col_id[p]/2];
     
        }
        temp.row16 = row_out[p] & 0x7FFF;
          temp.dummy = !((row_out[p] >> 15) & 1);
        temp.tileEnd = temp_flag.tileEnd;
        c_out[p] << temp;
      }
    }
    tK = lastTikeK ? 0 : tK + 1;
    i = lastTikeK ? i+1 : i;
  }
}



void Arbiter_C_10_1(int seq, tapa::istreams<float_vN, 10>& c_arb,
    tapa::ostream<float_vN>& c_ab) {
  float_vN temp_in;
  
  for (;;) {
    #pragma HLS PIPELINE II=1
    
    for (int i = 0; i < 10; i++) {
        temp_in = c_arb[i].read();
        c_ab << temp_in;
    }
  }
}


void Arbiter_C_8_4(int seq, tapa::istreams<float_vN, 8>& c_arb,
    tapa::ostreams<float_vB, NUM_C_CH>& c_ab) {
  
  for (;;) {
    #pragma HLS PIPELINE II=1
      float_vB temp_out[NUM_C_CH];
      #pragma HLS array_partition variable=temp_out complete
      for (int j = 0; j < 8; j++) {
        float_vN temp_in = c_arb[j].read();
        for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      for (int c = 0; c < NUM_C_CH; c++) {
        c_ab[c] << temp_out[c];
      }
    
  }
}

void Compute_C(int seq,tapa::istream<float_vB>& c_ab,
  tapa::istream<float_vB>& c_in,
  tapa::ostream<float_vB>& c_out,
  const float alpha, const float beta) {
  for(;;) {
    #pragma HLS pipeline II=1
    float_vB temp0, temp1, temp2;
    
    if (!c_ab.empty() && !c_in.empty()) {
      c_ab.try_read(temp0);
      c_in.try_read(temp1);
           
      for(int p = 0; p < B_CHUNK_SIZE; p++) 
        temp2[p] = alpha * temp0[p] + beta * temp1[p];
      c_out << temp2;
    }
  }
}



void hispmm(tapa::mmaps<uint64_v, NUM_A_CH> A,
          tapa::mmaps<float_vB, NUM_B_CH> B_in,
          tapa::mmaps<float_vB, NUM_C_CH> C_in,
          tapa::mmaps<float_vB, NUM_C_CH> C_out,
          const float alpha, const float beta,
          const uint32_t M, const uint32_t N, const uint32_t K,
          const uint32_t numTilesM, const uint32_t numTilesN, const uint32_t numTilesK,
          const uint32_t len, const uint32_t last_tile_idx, const uint16_t rp_time)
{
  tapa::streams<uint64_v2, NUM_PES_HALF, FIFO_DEPTH> FIFO_A_IN("a_in");
  tapa::streams<float_vB, ((NUM_PEG + 1)* NUM_B_CH), FIFO_DEPTH> FIFO_B_IN("b_in");
  tapa::streams<float_vN, NUM_PES, FIFO_LARGE_DEPTH> FIFO_C_ARB("c_arb");
  tapa::streams<float_vB, NUM_C_CH, FIFO_LARGE_DEPTH> FIFO_C_AB("c_ab");
  tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_IN("c_in");
  tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_OUT("c_out");
  tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
  tapa::streams<float_vN, 8, FIFO_LARGE_DEPTH> FIFO_C_AB_INTER("c_ab_inter");
 

    tapa::task()
        .invoke<tapa::join, NUM_A_CH>(MM2S_A, A, FIFO_A_IN, len, numTilesN, rp_time)
        .invoke<tapa::join, NUM_B_CH>(MM2S_B, B_in, FIFO_B_IN, numTilesM, numTilesN, numTilesK, K, rp_time)
        .invoke<tapa::join, NUM_C_CH>(MM2S_C, C_in, FIFO_C_IN, numTilesM, numTilesN, M, N, rp_time)
        .invoke<tapa::join, NUM_PEG>(PEG, FIFO_A_IN, FIFO_B_IN, FIFO_B_IN, FIFO_C_SHF, K, numTilesK, last_tile_idx)
        .invoke<tapa::detach, NUM_B_CH>(DummyRead, FIFO_B_IN)
        .invoke<tapa::join, NUM_PES>(Accumulator, FIFO_C_ARB, FIFO_C_SHF, M, numTilesM, last_tile_idx)
        .invoke<tapa::detach, 8>(Arbiter_C_10_1, tapa::seq(), FIFO_C_ARB, FIFO_C_AB_INTER)
        .invoke<tapa::detach>(Arbiter_C_8_4, tapa::seq(), FIFO_C_AB_INTER, FIFO_C_AB)
        .invoke<tapa::detach, NUM_C_CH>(Compute_C, tapa::seq(), FIFO_C_AB, FIFO_C_IN, FIFO_C_OUT, alpha, beta)
        .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, C_out, numTilesM, numTilesN, M, N, rp_time)
        ;
}
