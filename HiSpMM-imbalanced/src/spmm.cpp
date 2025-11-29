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

void PEG(tapa::istreams<uint64_v2, GROUP_SIZE/2>& a_in, 
  tapa::istreams<float_vB, NUM_B_CH>& b_in,
  tapa::ostreams<float_vB, NUM_B_CH>& b_out,
  tapa::ostreams<Cnoc_pkt, GROUP_SIZE>& c_out,
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
      temp_flag.sharedRow = (temp_in[0][0] >> 45) & 1;
      temp_flag.tileEnd = tileEnd & lastTikeK;
      temp_flag.last = last;
      
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
        Cnoc_pkt temp;
      #pragma HLS UNROLL
        for (int n = 0; n < N0; n++) {
          temp.val[n] = val_in[p] * buff_B[p/2][n][col_id[p]%2][col_id[p]/2];
     
        }
         
        if (temp_flag.sharedRow) {
          if (p % 2) {
            temp.row16 = row_out[p] & 0x7FFF;
            temp.dummy = !((row_out[p] >> 15) & 1);
            temp.bank = (uint8_t)(row_out[p-1] & ((1 << LOG_2_NUM_PES) - 1));
          }
          else {
            temp.row16 = row_out[p+1] & 0x7FFF;
            temp.dummy = !((row_out[p+1] >> 15) & 1);
            temp.bank = (uint8_t)(row_out[p] & ((1 << LOG_2_NUM_PES) - 1));
          }
        }
        else {
          temp.row16 = row_out[p] & 0x7FFF;
          temp.dummy = !((row_out[p] >> 15) & 1);
          temp.bank = 0;
        }
         
        temp.last = temp_flag.last;
        temp.tileEnd = temp_flag.tileEnd;
        temp.sharedRow = temp_flag.sharedRow;
        c_out[p] << temp;
      }
    }
    tK = lastTikeK ? 0 : tK + 1;
    i = lastTikeK ? i+1 : i;
    
  }
}

void DummyARB(tapa::istreams<float_vN, NUM_PES>& c_arb)
{
    for (;;) {
      #pragma HLS PIPELINE II=1
      for (int p = 0; p < NUM_PES; p++) {
        float_vN temp = c_arb[p].read();
      }
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
     
    init_c:for(int m = 0; m < read_len; m++) 
      #pragma HLS PIPELINE II=1
      for(int n = 0; n < N0; n++) 
        buffer_C[m][n] = 0;
      
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



void Arbiter_C_16_1(tapa::istreams<float_vN, 16>& c_arb,
    tapa::ostream<float_vB>& c_ab) {
  float_vB temp_out;
  float_vN temp_in;
  for (;;) {
    #pragma HLS PIPELINE II=1
   {
    loop_p_0: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[j].read();
      loop_p_0_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_0_write: c_ab << temp_out;
    
   } 
   
    #pragma HLS PIPELINE II=1
   {
    loop_p_1: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[2+j].read();
      loop_p_1_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_1_write: c_ab << temp_out;
   } 
   
    #pragma HLS PIPELINE II=1
   {
    loop_p_2: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[4+j].read();
      loop_p_2_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_2_write: c_ab << temp_out;
   } 
   
    #pragma HLS PIPELINE II=1
   {
    loop_p_3: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[6+j].read();
      loop_p_3_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_3_write: c_ab << temp_out;
   }
   #pragma HLS PIPELINE II=1
   {
    loop_p_4: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[8+j].read();
      loop_p_4_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_4_write: c_ab << temp_out;
   }
   #pragma HLS PIPELINE II=1
   {
    loop_p_5: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[10+j].read();
      loop_p_5_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_5_write: c_ab << temp_out;
   }
   
   #pragma HLS PIPELINE II=1
   {
    loop_p_6: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[12+j].read();
      loop_p_6_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_6_write: c_ab << temp_out;
   }
   
   #pragma HLS PIPELINE II=1
   {
    loop_p_7: for (int j = 0; j < 2; j++) {
      temp_in = c_arb[14+j].read();
      loop_p_7_k: for (int k = 0; k < N0; k++) {
        temp_out[(j%2)*N0 + k] = temp_in[k];
      }
    }
    p_7_write: c_ab << temp_out;
   }    
   
   
  }
}

void Arbiter_C(tapa::istreams<float_vN, NUM_PES>& c_arb,
    tapa::ostreams<float_vB, NUM_C_CH>& c_ab) {
  float_vB temp_out[NUM_C_CH];
  float_vN temp_in;
  // #pragma HLS ARRAY_PARTITION variable=temp_out complete dim=1
  // #pragma HLS BIND_STORAGE variable=temp_out type=RAM_S2P impl=bram
  
  for (;;) {
    #pragma HLS PIPELINE II=1
    {
      loop_p_0: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[0 + j].read();
        loop_p_0_k: for (int k = 0; k < N0; k++) {
            temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_0_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
    
    #pragma HLS PIPELINE II=1
    {
      loop_p_1: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*2 + j].read();
        loop_p_1_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_1_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
  
    #pragma HLS PIPELINE II=1
    {
      loop_p_2: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*4 + j].read();
        loop_p_2_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_2_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }

    #pragma HLS PIPELINE II=1
    {
      loop_p_3: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*6 + j].read();
        loop_p_3_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_3_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
    #pragma HLS PIPELINE II=1
    {
      loop_p_4: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*8 + j].read();
        loop_p_4_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_4_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
    #pragma HLS PIPELINE II=1
    {
      loop_p_5: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*10 + j].read();
        loop_p_5_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_5_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
    #pragma HLS PIPELINE II=1
    {
      loop_p_6: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*12 + j].read();
        loop_p_6_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_6_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
    #pragma HLS PIPELINE II=1
    {
      loop_p_7: for (int j = 0; j < NUM_C_CH*2; j++) {
        temp_in = c_arb[NUM_C_CH*14 + j].read();
        loop_p_7_k: for (int k = 0; k < N0; k++) {
          temp_out[j/2][(j%2)*N0 + k] = temp_in[k];
        }
      }
      loop_p_7_write: for (int c = 0; c < NUM_C_CH; c++)
        c_ab[c] << temp_out[c];
    }
  }
}

void DummyPE(tapa::istream<uint64_v2>& a_in) {
  for (;;) {
  #pragma HLS PIPELINE II=1
      uint64_v2 temp = a_in.read();
  }
}

void DummyComp(tapa::istream<float_vN>& c_val,
  tapa::istream<uint16_t>& c_row,
  tapa::istream<flags_pkt>& c_flag) {
  for (;;) {
  #pragma HLS PIPELINE II=1
      float_vN temp_val = c_val.read();
      uint16_t temp_row = c_row.read();
      flags_pkt temp_flags = c_flag.read();
  }
}

void DummyC(tapa::istream<float_vB>& c_in) {
  for (;;) {
  #pragma HLS PIPELINE II=1
      float_vB temp_val = c_in.read();
  }
}

void Compute_C(tapa::istream<float_vB>& c_ab,
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

void ADD_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {

    float_vN temp[2];
    bool dummy[2];
    float_vN sum;

    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        for (int p = 0; p < 8; p++)
          sum[p] = curr_in0.val[p] + curr_in1.val[p];

        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy))
        {
          for (int p = 0; p < 8; p++){
            temp[0][p] = sum[p];
            temp[1][p] = (float)0;
          }

          dummy[0] = false;
          dummy[1] = true;
        }

        else {
          for (int p = 0; p < 8; p++) {
            temp[0][p] = curr_in0.val[p];
            temp[1][p] = curr_in1.val[p];
          }

          dummy[0] = curr_in0.dummy;
          dummy[1] = curr_in1.dummy;
        }

        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;

        for (int p = 0; p < N0; p++) {
          curr_out0.val[p] = temp[0][p];
          curr_out1.val[p] = temp[1][p];
        }

        c_out0 << curr_out0;
        c_out1 << curr_out1;

        last = curr_in0.last & curr_in1.last;    
    }
}

void ADD_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    float_vN temp[2];
    bool dummy[2];
    float_vN sum;
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
      
        for (int p = 0; p < N0; p++) // ---> Cnoc_pkt.val is a float val[N0]
          sum[p] = curr_in0.val[p] + curr_in1.val[p];

        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy))
        {
          for (int p = 0; p < N0; p++){ // ---> Cnoc_pkt.val is a float val[N0]
            temp[1][p] = sum[p];
            temp[0][p] = (float)0;
          }

          dummy[1] = false;
          dummy[0] = true;
        }

        else {
          for (int p = 0; p < N0; p++) {
            temp[0][p] = curr_in0.val[p];
            temp[1][p] = curr_in1.val[p];
          }

          dummy[0] = curr_in0.dummy;
          dummy[1] = curr_in1.dummy;
        }

        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0]; // everything else is directly transferred
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        for (int p = 0; p < N0; p++) { // ---> Cnoc_pkt.val is a float val[N0]
          curr_out0.val[p] = temp[0][p];
          curr_out1.val[p] = temp[1][p];
        }
        c_out0 << curr_out0;
        c_out1 << curr_out1;
        
        last = curr_in0.last & curr_in1.last;    
    }
}

void ADD_X(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    float_vN temp[2];
    bool dummy[2];
    float_vN sum;

    for(bool last = false; !last; ) {
        #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        for (int p = 0; p < N0; p++) // ---> Cnoc_pkt.val is a float val[N0]
          sum[p] = curr_in0.val[p] + curr_in1.val[p];
        
        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy)) // ---> Cnoc_pkt.sharedRow is a bool
        {
          bool i = ((curr_in0.bank >> (LOG_2_NUM_PES - 1)) & 1);
          for (int p = 0; p < N0; p++){ // ---> Cnoc_pkt.val is a float val[N0]
            temp[i][p] = sum[p];
            temp[!i][p] = (float)0;
          }

          dummy[i] = false;
          dummy[!i] = true;
        }

        else {
          for (int p = 0; p < N0; p++) { // ---> Cnoc_pkt.val is a float val[N0]
            temp[0][p] = curr_in0.val[p];
            temp[1][p] = curr_in1.val[p];
          }

          dummy[0] = curr_in0.dummy;
          dummy[1] = curr_in1.dummy;
        }
        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        for (int p = 0; p < N0; p++) { // ---> Cnoc_pkt.val is a float val[N0]
          curr_out0.val[p] = temp[0][p];
          curr_out1.val[p] = temp[1][p];
        }
        c_out0 << curr_out0;
        c_out1 << curr_out1;

        last = curr_in0.last & curr_in1.last;
    }  
}

void SWB0_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cvec_pkt>& c_out0, tapa::ostream<Cvec_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cvec_pkt curr_out[2];

        bool i = (curr_in0.bank & 1) && curr_in0.sharedRow;
        curr_out[i].dummy = curr_in0.dummy;
        curr_out[!i].dummy = curr_in1.dummy;
        curr_out[i].tileEnd = curr_in0.tileEnd;
        curr_out[!i].tileEnd = curr_in1.tileEnd;
        curr_out[i].row16 = curr_in0.row16;
        curr_out[!i].row16 = curr_in1.row16;
        for (int p = 0; p < N0; p++) {
          curr_out[i].val[p] = curr_in0.val[p];
          curr_out[!i].val[p] = curr_in1.val[p];
        }

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

void SWB1_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cvec_pkt>& c_out0, tapa::ostream<Cvec_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cvec_pkt curr_out[2];


        bool i = (curr_in1.bank & 1) || (!curr_in1.sharedRow);
        curr_out[i].dummy = curr_in1.dummy;
        curr_out[!i].dummy = curr_in0.dummy;
        curr_out[i].tileEnd = curr_in1.tileEnd;
        curr_out[!i].tileEnd = curr_in0.tileEnd;
        curr_out[i].row16 = curr_in1.row16;
        curr_out[!i].row16 = curr_in0.row16;

        for (int p = 0; p < N0; p++) {
          curr_out[i].val[p] = curr_in1.val[p];
          curr_out[!i].val[p] = curr_in0.val[p];
        }

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

template<int n>
void SWB0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];
        bool i = ((curr_in0.bank >> n) & 1) && (curr_in0.sharedRow);
        curr_out[i] = curr_in0;
        curr_out[!i] = curr_in1;
        c_out0 << curr_out[0];
        c_out1 << curr_out[1];
        last = curr_in0.last & curr_in1.last;    
    }
}

template<int n>
void SWB1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];


        bool i = ((curr_in1.bank >> n) & 1) || (!curr_in1.sharedRow);
        curr_out[i] = curr_in1;
        curr_out[!i] = curr_in0;

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

void SSW(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];

        bool i = curr_in0.sharedRow;

        curr_out[i] = curr_in0;
        curr_out[!i] = curr_in1;


        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

void SWB0_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<1>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<1>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_2(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<2>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_2(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<2>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_3(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<3>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_3(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<3>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_4(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<4>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_4(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<4>(c_in0, c_in1, c_out0, c_out1);
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
  tapa::streams<float_vN, NUM_PES, FIFO_DEPTH> FIFO_C_ARB("c_arb");
  tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_AB("c_ab");
  tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_IN("c_in");
  tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_OUT("c_out");
  tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
  tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_BUF("c_buf");

tapa::stream<Cnoc_pkt, 72> s_0_0("s_0_0");
tapa::stream<Cnoc_pkt, 2> s_1_0("s_1_0");
tapa::stream<Cnoc_pkt, 2> s_2_0("s_2_0");
tapa::stream<Cnoc_pkt, 10> s_3_0("s_3_0");
tapa::stream<Cnoc_pkt, 10> s_4_0("s_4_0");
tapa::stream<Cnoc_pkt, 2> s_5_0("s_5_0");
tapa::stream<Cnoc_pkt, 2> s_6_0("s_6_0");
tapa::stream<Cnoc_pkt, 20> s_7_0("s_7_0");
tapa::stream<Cnoc_pkt, 20> s_8_0("s_8_0");
tapa::stream<Cnoc_pkt, 2> s_9_0("s_9_0");
tapa::stream<Cnoc_pkt, 2> s_10_0("s_10_0");
tapa::stream<Cnoc_pkt, 10> s_11_0("s_11_0");
tapa::stream<Cnoc_pkt, 10> s_12_0("s_12_0");
tapa::stream<Cnoc_pkt, 2> s_13_0("s_13_0");
tapa::stream<Cnoc_pkt, 2> s_14_0("s_14_0");
tapa::stream<Cnoc_pkt, 30> s_15_0("s_15_0");
tapa::stream<Cnoc_pkt, 30> s_16_0("s_16_0");
tapa::stream<Cnoc_pkt, 2> s_17_0("s_17_0");
tapa::stream<Cnoc_pkt, 2> s_18_0("s_18_0");
tapa::stream<Cnoc_pkt, 10> s_19_0("s_19_0");
tapa::stream<Cnoc_pkt, 10> s_20_0("s_20_0");
tapa::stream<Cnoc_pkt, 2> s_21_0("s_21_0");
tapa::stream<Cnoc_pkt, 2> s_22_0("s_22_0");
tapa::stream<Cnoc_pkt, 20> s_23_0("s_23_0");
tapa::stream<Cnoc_pkt, 20> s_24_0("s_24_0");
tapa::stream<Cnoc_pkt, 2> s_25_0("s_25_0");
tapa::stream<Cnoc_pkt, 2> s_26_0("s_26_0");
tapa::stream<Cnoc_pkt, 10> s_27_0("s_27_0");
tapa::stream<Cnoc_pkt, 10> s_28_0("s_28_0");
tapa::stream<Cnoc_pkt, 2> s_29_0("s_29_0");
tapa::stream<Cnoc_pkt, 2> s_30_0("s_30_0");
tapa::stream<Cnoc_pkt, 40> s_31_0("s_31_0");
tapa::stream<Cnoc_pkt, 40> s_32_0("s_32_0");
tapa::stream<Cnoc_pkt, 2> s_33_0("s_33_0");
tapa::stream<Cnoc_pkt, 2> s_34_0("s_34_0");
tapa::stream<Cnoc_pkt, 10> s_35_0("s_35_0");
tapa::stream<Cnoc_pkt, 10> s_36_0("s_36_0");
tapa::stream<Cnoc_pkt, 2> s_37_0("s_37_0");
tapa::stream<Cnoc_pkt, 2> s_38_0("s_38_0");
tapa::stream<Cnoc_pkt, 20> s_39_0("s_39_0");
tapa::stream<Cnoc_pkt, 20> s_40_0("s_40_0");
tapa::stream<Cnoc_pkt, 2> s_41_0("s_41_0");
tapa::stream<Cnoc_pkt, 2> s_42_0("s_42_0");
tapa::stream<Cnoc_pkt, 10> s_43_0("s_43_0");
tapa::stream<Cnoc_pkt, 10> s_44_0("s_44_0");
tapa::stream<Cnoc_pkt, 2> s_45_0("s_45_0");
tapa::stream<Cnoc_pkt, 2> s_46_0("s_46_0");
tapa::stream<Cnoc_pkt, 30> s_47_0("s_47_0");
tapa::stream<Cnoc_pkt, 30> s_48_0("s_48_0");
tapa::stream<Cnoc_pkt, 2> s_49_0("s_49_0");
tapa::stream<Cnoc_pkt, 2> s_50_0("s_50_0");
tapa::stream<Cnoc_pkt, 10> s_51_0("s_51_0");
tapa::stream<Cnoc_pkt, 10> s_52_0("s_52_0");
tapa::stream<Cnoc_pkt, 2> s_53_0("s_53_0");
tapa::stream<Cnoc_pkt, 2> s_54_0("s_54_0");
tapa::stream<Cnoc_pkt, 20> s_55_0("s_55_0");
tapa::stream<Cnoc_pkt, 20> s_56_0("s_56_0");
tapa::stream<Cnoc_pkt, 2> s_57_0("s_57_0");
tapa::stream<Cnoc_pkt, 2> s_58_0("s_58_0");
tapa::stream<Cnoc_pkt, 10> s_59_0("s_59_0");
tapa::stream<Cnoc_pkt, 10> s_60_0("s_60_0");
tapa::stream<Cnoc_pkt, 2> s_61_0("s_61_0");
tapa::stream<Cnoc_pkt, 2> s_62_0("s_62_0");
tapa::stream<Cnoc_pkt, 72> s_63_0("s_63_0");
tapa::stream<Cnoc_pkt, 68> s_1_1("s_1_1");
tapa::stream<Cnoc_pkt, 8> s_2_1("s_2_1");
tapa::stream<Cnoc_pkt, 8> s_5_1("s_5_1");
tapa::stream<Cnoc_pkt, 68> s_6_1("s_6_1");
tapa::stream<Cnoc_pkt, 68> s_9_1("s_9_1");
tapa::stream<Cnoc_pkt, 8> s_10_1("s_10_1");
tapa::stream<Cnoc_pkt, 8> s_13_1("s_13_1");
tapa::stream<Cnoc_pkt, 68> s_14_1("s_14_1");
tapa::stream<Cnoc_pkt, 68> s_17_1("s_17_1");
tapa::stream<Cnoc_pkt, 8> s_18_1("s_18_1");
tapa::stream<Cnoc_pkt, 8> s_21_1("s_21_1");
tapa::stream<Cnoc_pkt, 68> s_22_1("s_22_1");
tapa::stream<Cnoc_pkt, 68> s_25_1("s_25_1");
tapa::stream<Cnoc_pkt, 8> s_26_1("s_26_1");
tapa::stream<Cnoc_pkt, 8> s_29_1("s_29_1");
tapa::stream<Cnoc_pkt, 68> s_30_1("s_30_1");
tapa::stream<Cnoc_pkt, 68> s_33_1("s_33_1");
tapa::stream<Cnoc_pkt, 8> s_34_1("s_34_1");
tapa::stream<Cnoc_pkt, 8> s_37_1("s_37_1");
tapa::stream<Cnoc_pkt, 68> s_38_1("s_38_1");
tapa::stream<Cnoc_pkt, 68> s_41_1("s_41_1");
tapa::stream<Cnoc_pkt, 8> s_42_1("s_42_1");
tapa::stream<Cnoc_pkt, 8> s_45_1("s_45_1");
tapa::stream<Cnoc_pkt, 68> s_46_1("s_46_1");
tapa::stream<Cnoc_pkt, 68> s_49_1("s_49_1");
tapa::stream<Cnoc_pkt, 8> s_50_1("s_50_1");
tapa::stream<Cnoc_pkt, 8> s_53_1("s_53_1");
tapa::stream<Cnoc_pkt, 68> s_54_1("s_54_1");
tapa::stream<Cnoc_pkt, 68> s_57_1("s_57_1");
tapa::stream<Cnoc_pkt, 8> s_58_1("s_58_1");
tapa::stream<Cnoc_pkt, 8> s_61_1("s_61_1");
tapa::stream<Cnoc_pkt, 68> s_62_1("s_62_1");
tapa::stream<Cnoc_pkt, 58> s_2_2("s_2_2");
tapa::stream<Cnoc_pkt, 2> s_3_1("s_3_1");
tapa::stream<Cnoc_pkt, 2> s_4_1("s_4_1");
tapa::stream<Cnoc_pkt, 58> s_5_2("s_5_2");
tapa::stream<Cnoc_pkt, 54> s_3_2("s_3_2");
tapa::stream<Cnoc_pkt, 8> s_4_2("s_4_2");
tapa::stream<Cnoc_pkt, 58> s_10_2("s_10_2");
tapa::stream<Cnoc_pkt, 2> s_11_1("s_11_1");
tapa::stream<Cnoc_pkt, 2> s_12_1("s_12_1");
tapa::stream<Cnoc_pkt, 58> s_13_2("s_13_2");
tapa::stream<Cnoc_pkt, 8> s_11_2("s_11_2");
tapa::stream<Cnoc_pkt, 54> s_12_2("s_12_2");
tapa::stream<Cnoc_pkt, 58> s_18_2("s_18_2");
tapa::stream<Cnoc_pkt, 2> s_19_1("s_19_1");
tapa::stream<Cnoc_pkt, 2> s_20_1("s_20_1");
tapa::stream<Cnoc_pkt, 58> s_21_2("s_21_2");
tapa::stream<Cnoc_pkt, 54> s_19_2("s_19_2");
tapa::stream<Cnoc_pkt, 8> s_20_2("s_20_2");
tapa::stream<Cnoc_pkt, 58> s_26_2("s_26_2");
tapa::stream<Cnoc_pkt, 2> s_27_1("s_27_1");
tapa::stream<Cnoc_pkt, 2> s_28_1("s_28_1");
tapa::stream<Cnoc_pkt, 58> s_29_2("s_29_2");
tapa::stream<Cnoc_pkt, 8> s_27_2("s_27_2");
tapa::stream<Cnoc_pkt, 54> s_28_2("s_28_2");
tapa::stream<Cnoc_pkt, 58> s_34_2("s_34_2");
tapa::stream<Cnoc_pkt, 2> s_35_1("s_35_1");
tapa::stream<Cnoc_pkt, 2> s_36_1("s_36_1");
tapa::stream<Cnoc_pkt, 58> s_37_2("s_37_2");
tapa::stream<Cnoc_pkt, 54> s_35_2("s_35_2");
tapa::stream<Cnoc_pkt, 8> s_36_2("s_36_2");
tapa::stream<Cnoc_pkt, 58> s_42_2("s_42_2");
tapa::stream<Cnoc_pkt, 2> s_43_1("s_43_1");
tapa::stream<Cnoc_pkt, 2> s_44_1("s_44_1");
tapa::stream<Cnoc_pkt, 58> s_45_2("s_45_2");
tapa::stream<Cnoc_pkt, 8> s_43_2("s_43_2");
tapa::stream<Cnoc_pkt, 54> s_44_2("s_44_2");
tapa::stream<Cnoc_pkt, 58> s_50_2("s_50_2");
tapa::stream<Cnoc_pkt, 2> s_51_1("s_51_1");
tapa::stream<Cnoc_pkt, 2> s_52_1("s_52_1");
tapa::stream<Cnoc_pkt, 58> s_53_2("s_53_2");
tapa::stream<Cnoc_pkt, 54> s_51_2("s_51_2");
tapa::stream<Cnoc_pkt, 8> s_52_2("s_52_2");
tapa::stream<Cnoc_pkt, 58> s_58_2("s_58_2");
tapa::stream<Cnoc_pkt, 2> s_59_1("s_59_1");
tapa::stream<Cnoc_pkt, 2> s_60_1("s_60_1");
tapa::stream<Cnoc_pkt, 58> s_61_2("s_61_2");
tapa::stream<Cnoc_pkt, 8> s_59_2("s_59_2");
tapa::stream<Cnoc_pkt, 54> s_60_2("s_60_2");
tapa::stream<Cnoc_pkt, 44> s_4_3("s_4_3");
tapa::stream<Cnoc_pkt, 2> s_7_1("s_7_1");
tapa::stream<Cnoc_pkt, 2> s_8_1("s_8_1");
tapa::stream<Cnoc_pkt, 44> s_11_3("s_11_3");
tapa::stream<Cnoc_pkt, 40> s_7_2("s_7_2");
tapa::stream<Cnoc_pkt, 8> s_8_2("s_8_2");
tapa::stream<Cnoc_pkt, 44> s_20_3("s_20_3");
tapa::stream<Cnoc_pkt, 2> s_23_1("s_23_1");
tapa::stream<Cnoc_pkt, 2> s_24_1("s_24_1");
tapa::stream<Cnoc_pkt, 44> s_27_3("s_27_3");
tapa::stream<Cnoc_pkt, 8> s_23_2("s_23_2");
tapa::stream<Cnoc_pkt, 40> s_24_2("s_24_2");
tapa::stream<Cnoc_pkt, 44> s_36_3("s_36_3");
tapa::stream<Cnoc_pkt, 2> s_39_1("s_39_1");
tapa::stream<Cnoc_pkt, 2> s_40_1("s_40_1");
tapa::stream<Cnoc_pkt, 44> s_43_3("s_43_3");
tapa::stream<Cnoc_pkt, 40> s_39_2("s_39_2");
tapa::stream<Cnoc_pkt, 8> s_40_2("s_40_2");
tapa::stream<Cnoc_pkt, 44> s_52_3("s_52_3");
tapa::stream<Cnoc_pkt, 2> s_55_1("s_55_1");
tapa::stream<Cnoc_pkt, 2> s_56_1("s_56_1");
tapa::stream<Cnoc_pkt, 44> s_59_3("s_59_3");
tapa::stream<Cnoc_pkt, 8> s_55_2("s_55_2");
tapa::stream<Cnoc_pkt, 40> s_56_2("s_56_2");
tapa::stream<Cnoc_pkt, 30> s_8_3("s_8_3");
tapa::stream<Cnoc_pkt, 2> s_15_1("s_15_1");
tapa::stream<Cnoc_pkt, 2> s_16_1("s_16_1");
tapa::stream<Cnoc_pkt, 30> s_23_3("s_23_3");
tapa::stream<Cnoc_pkt, 20> s_15_2("s_15_2");
tapa::stream<Cnoc_pkt, 8> s_16_2("s_16_2");
tapa::stream<Cnoc_pkt, 30> s_40_3("s_40_3");
tapa::stream<Cnoc_pkt, 2> s_47_1("s_47_1");
tapa::stream<Cnoc_pkt, 2> s_48_1("s_48_1");
tapa::stream<Cnoc_pkt, 30> s_55_3("s_55_3");
tapa::stream<Cnoc_pkt, 8> s_47_2("s_47_2");
tapa::stream<Cnoc_pkt, 20> s_48_2("s_48_2");
tapa::stream<Cnoc_pkt, 10> s_16_3("s_16_3");
tapa::stream<Cnoc_pkt, 2> s_31_1("s_31_1");
tapa::stream<Cnoc_pkt, 2> s_32_1("s_32_1");
tapa::stream<Cnoc_pkt, 10> s_47_3("s_47_3");
tapa::stream<Cnoc_pkt, 8> s_31_2("s_31_2");
tapa::stream<Cnoc_pkt, 8> s_32_2("s_32_2");
tapa::stream<Cnoc_pkt, 2> s_16_4("s_16_4");
tapa::stream<Cnoc_pkt, 22> s_31_3("s_31_3");
tapa::stream<Cnoc_pkt, 22> s_32_3("s_32_3");
tapa::stream<Cnoc_pkt, 2> s_47_4("s_47_4");
tapa::stream<Cnoc_pkt, 8> s_15_3("s_15_3");
tapa::stream<Cnoc_pkt, 8> s_16_5("s_16_5");
tapa::stream<Cnoc_pkt, 2> s_8_4("s_8_4");
tapa::stream<Cnoc_pkt, 12> s_15_4("s_15_4");
tapa::stream<Cnoc_pkt, 12> s_16_6("s_16_6");
tapa::stream<Cnoc_pkt, 2> s_23_4("s_23_4");
tapa::stream<Cnoc_pkt, 8> s_47_5("s_47_5");
tapa::stream<Cnoc_pkt, 8> s_48_3("s_48_3");
tapa::stream<Cnoc_pkt, 2> s_40_4("s_40_4");
tapa::stream<Cnoc_pkt, 12> s_47_6("s_47_6");
tapa::stream<Cnoc_pkt, 12> s_48_4("s_48_4");
tapa::stream<Cnoc_pkt, 2> s_55_4("s_55_4");
tapa::stream<Cnoc_pkt, 2> s_7_3("s_7_3");
tapa::stream<Cnoc_pkt, 2> s_8_5("s_8_5");
tapa::stream<Cnoc_pkt, 2> s_4_4("s_4_4");
tapa::stream<Cnoc_pkt, 8> s_7_4("s_7_4");
tapa::stream<Cnoc_pkt, 8> s_8_6("s_8_6");
tapa::stream<Cnoc_pkt, 2> s_11_4("s_11_4");
tapa::stream<Cnoc_pkt, 2> s_23_5("s_23_5");
tapa::stream<Cnoc_pkt, 2> s_24_3("s_24_3");
tapa::stream<Cnoc_pkt, 2> s_20_4("s_20_4");
tapa::stream<Cnoc_pkt, 8> s_23_6("s_23_6");
tapa::stream<Cnoc_pkt, 8> s_24_4("s_24_4");
tapa::stream<Cnoc_pkt, 2> s_27_4("s_27_4");
tapa::stream<Cnoc_pkt, 2> s_39_3("s_39_3");
tapa::stream<Cnoc_pkt, 2> s_40_5("s_40_5");
tapa::stream<Cnoc_pkt, 2> s_36_4("s_36_4");
tapa::stream<Cnoc_pkt, 8> s_39_4("s_39_4");
tapa::stream<Cnoc_pkt, 8> s_40_6("s_40_6");
tapa::stream<Cnoc_pkt, 2> s_43_4("s_43_4");
tapa::stream<Cnoc_pkt, 2> s_55_5("s_55_5");
tapa::stream<Cnoc_pkt, 2> s_56_3("s_56_3");
tapa::stream<Cnoc_pkt, 2> s_52_4("s_52_4");
tapa::stream<Cnoc_pkt, 8> s_55_6("s_55_6");
tapa::stream<Cnoc_pkt, 8> s_56_4("s_56_4");
tapa::stream<Cnoc_pkt, 2> s_59_4("s_59_4");
tapa::stream<Cnoc_pkt, 2> s_3_3("s_3_3");
tapa::stream<Cnoc_pkt, 2> s_4_5("s_4_5");
tapa::stream<Cnoc_pkt, 2> s_2_3("s_2_3");
tapa::stream<Cnoc_pkt, 4> s_3_4("s_3_4");
tapa::stream<Cnoc_pkt, 4> s_4_6("s_4_6");
tapa::stream<Cnoc_pkt, 2> s_5_3("s_5_3");
tapa::stream<Cnoc_pkt, 2> s_11_5("s_11_5");
tapa::stream<Cnoc_pkt, 2> s_12_3("s_12_3");
tapa::stream<Cnoc_pkt, 2> s_10_3("s_10_3");
tapa::stream<Cnoc_pkt, 4> s_11_6("s_11_6");
tapa::stream<Cnoc_pkt, 4> s_12_4("s_12_4");
tapa::stream<Cnoc_pkt, 2> s_13_3("s_13_3");
tapa::stream<Cnoc_pkt, 2> s_19_3("s_19_3");
tapa::stream<Cnoc_pkt, 2> s_20_5("s_20_5");
tapa::stream<Cnoc_pkt, 2> s_18_3("s_18_3");
tapa::stream<Cnoc_pkt, 4> s_19_4("s_19_4");
tapa::stream<Cnoc_pkt, 4> s_20_6("s_20_6");
tapa::stream<Cnoc_pkt, 2> s_21_3("s_21_3");
tapa::stream<Cnoc_pkt, 2> s_27_5("s_27_5");
tapa::stream<Cnoc_pkt, 2> s_28_3("s_28_3");
tapa::stream<Cnoc_pkt, 2> s_26_3("s_26_3");
tapa::stream<Cnoc_pkt, 4> s_27_6("s_27_6");
tapa::stream<Cnoc_pkt, 4> s_28_4("s_28_4");
tapa::stream<Cnoc_pkt, 2> s_29_3("s_29_3");
tapa::stream<Cnoc_pkt, 2> s_35_3("s_35_3");
tapa::stream<Cnoc_pkt, 2> s_36_5("s_36_5");
tapa::stream<Cnoc_pkt, 2> s_34_3("s_34_3");
tapa::stream<Cnoc_pkt, 4> s_35_4("s_35_4");
tapa::stream<Cnoc_pkt, 4> s_36_6("s_36_6");
tapa::stream<Cnoc_pkt, 2> s_37_3("s_37_3");
tapa::stream<Cnoc_pkt, 2> s_43_5("s_43_5");
tapa::stream<Cnoc_pkt, 2> s_44_3("s_44_3");
tapa::stream<Cnoc_pkt, 2> s_42_3("s_42_3");
tapa::stream<Cnoc_pkt, 4> s_43_6("s_43_6");
tapa::stream<Cnoc_pkt, 4> s_44_4("s_44_4");
tapa::stream<Cnoc_pkt, 2> s_45_3("s_45_3");
tapa::stream<Cnoc_pkt, 2> s_51_3("s_51_3");
tapa::stream<Cnoc_pkt, 2> s_52_5("s_52_5");
tapa::stream<Cnoc_pkt, 2> s_50_3("s_50_3");
tapa::stream<Cnoc_pkt, 4> s_51_4("s_51_4");
tapa::stream<Cnoc_pkt, 4> s_52_6("s_52_6");
tapa::stream<Cnoc_pkt, 2> s_53_3("s_53_3");
tapa::stream<Cnoc_pkt, 2> s_59_5("s_59_5");
tapa::stream<Cnoc_pkt, 2> s_60_3("s_60_3");
tapa::stream<Cnoc_pkt, 2> s_58_3("s_58_3");
tapa::stream<Cnoc_pkt, 4> s_59_6("s_59_6");
tapa::stream<Cnoc_pkt, 4> s_60_4("s_60_4");
tapa::stream<Cnoc_pkt, 2> s_61_3("s_61_3");
tapa::stream<Cnoc_pkt, 2> s_1_2("s_1_2");
tapa::stream<Cnoc_pkt, 2> s_2_4("s_2_4");
tapa::stream<Cnoc_pkt, 2> s_5_4("s_5_4");
tapa::stream<Cnoc_pkt, 2> s_6_2("s_6_2");
tapa::stream<Cnoc_pkt, 2> s_9_2("s_9_2");
tapa::stream<Cnoc_pkt, 2> s_10_4("s_10_4");
tapa::stream<Cnoc_pkt, 2> s_13_4("s_13_4");
tapa::stream<Cnoc_pkt, 2> s_14_2("s_14_2");
tapa::stream<Cnoc_pkt, 2> s_17_2("s_17_2");
tapa::stream<Cnoc_pkt, 2> s_18_4("s_18_4");
tapa::stream<Cnoc_pkt, 2> s_21_4("s_21_4");
tapa::stream<Cnoc_pkt, 2> s_22_2("s_22_2");
tapa::stream<Cnoc_pkt, 2> s_25_2("s_25_2");
tapa::stream<Cnoc_pkt, 2> s_26_4("s_26_4");
tapa::stream<Cnoc_pkt, 2> s_29_4("s_29_4");
tapa::stream<Cnoc_pkt, 2> s_30_2("s_30_2");
tapa::stream<Cnoc_pkt, 2> s_33_2("s_33_2");
tapa::stream<Cnoc_pkt, 2> s_34_4("s_34_4");
tapa::stream<Cnoc_pkt, 2> s_37_4("s_37_4");
tapa::stream<Cnoc_pkt, 2> s_38_2("s_38_2");
tapa::stream<Cnoc_pkt, 2> s_41_2("s_41_2");
tapa::stream<Cnoc_pkt, 2> s_42_4("s_42_4");
tapa::stream<Cnoc_pkt, 2> s_45_4("s_45_4");
tapa::stream<Cnoc_pkt, 2> s_46_2("s_46_2");
tapa::stream<Cnoc_pkt, 2> s_49_2("s_49_2");
tapa::stream<Cnoc_pkt, 2> s_50_4("s_50_4");
tapa::stream<Cnoc_pkt, 2> s_53_4("s_53_4");
tapa::stream<Cnoc_pkt, 2> s_54_2("s_54_2");
tapa::stream<Cnoc_pkt, 2> s_57_2("s_57_2");
tapa::stream<Cnoc_pkt, 2> s_58_4("s_58_4");
tapa::stream<Cnoc_pkt, 2> s_61_4("s_61_4");
tapa::stream<Cnoc_pkt, 2> s_62_2("s_62_2");


    tapa::task()
        .invoke<tapa::join, NUM_A_CH>(MM2S_A, A, FIFO_A_IN, len, numTilesN, rp_time)
        .invoke<tapa::join, NUM_B_CH>(MM2S_B, B_in, FIFO_B_IN, numTilesM, numTilesN, numTilesK, K, rp_time)
        .invoke<tapa::join, NUM_C_CH>(MM2S_C, C_in, FIFO_C_IN, numTilesM, numTilesN, M, N, rp_time)
        .invoke<tapa::join, NUM_PEG>(PEG, FIFO_A_IN, FIFO_B_IN, FIFO_B_IN, FIFO_C_SHF, K, numTilesK, last_tile_idx)
        .invoke<tapa::detach, NUM_B_CH>(DummyRead, FIFO_B_IN)
        .invoke(ADD_1, FIFO_C_SHF[0], FIFO_C_SHF[1], s_0_0, s_1_0)/*0*/
        .invoke(ADD_0, FIFO_C_SHF[2], FIFO_C_SHF[3], s_2_0, s_3_0)/*1*/
        .invoke(ADD_1, FIFO_C_SHF[4], FIFO_C_SHF[5], s_4_0, s_5_0)/*2*/
        .invoke(ADD_0, FIFO_C_SHF[6], FIFO_C_SHF[7], s_6_0, s_7_0)/*3*/
        .invoke(ADD_1, FIFO_C_SHF[8], FIFO_C_SHF[9], s_8_0, s_9_0)/*4*/
        .invoke(ADD_0, FIFO_C_SHF[10], FIFO_C_SHF[11], s_10_0, s_11_0)/*5*/
        .invoke(ADD_1, FIFO_C_SHF[12], FIFO_C_SHF[13], s_12_0, s_13_0)/*6*/
        .invoke(ADD_0, FIFO_C_SHF[14], FIFO_C_SHF[15], s_14_0, s_15_0)/*7*/
        .invoke(ADD_1, FIFO_C_SHF[16], FIFO_C_SHF[17], s_16_0, s_17_0)/*8*/
        .invoke(ADD_0, FIFO_C_SHF[18], FIFO_C_SHF[19], s_18_0, s_19_0)/*9*/
        .invoke(ADD_1, FIFO_C_SHF[20], FIFO_C_SHF[21], s_20_0, s_21_0)/*10*/
        .invoke(ADD_0, FIFO_C_SHF[22], FIFO_C_SHF[23], s_22_0, s_23_0)/*11*/
        .invoke(ADD_1, FIFO_C_SHF[24], FIFO_C_SHF[25], s_24_0, s_25_0)/*12*/
        .invoke(ADD_0, FIFO_C_SHF[26], FIFO_C_SHF[27], s_26_0, s_27_0)/*13*/
        .invoke(ADD_1, FIFO_C_SHF[28], FIFO_C_SHF[29], s_28_0, s_29_0)/*14*/
        .invoke(ADD_0, FIFO_C_SHF[30], FIFO_C_SHF[31], s_30_0, s_31_0)/*15*/
        .invoke(ADD_1, FIFO_C_SHF[32], FIFO_C_SHF[33], s_32_0, s_33_0)/*16*/
        .invoke(ADD_0, FIFO_C_SHF[34], FIFO_C_SHF[35], s_34_0, s_35_0)/*17*/
        .invoke(ADD_1, FIFO_C_SHF[36], FIFO_C_SHF[37], s_36_0, s_37_0)/*18*/
        .invoke(ADD_0, FIFO_C_SHF[38], FIFO_C_SHF[39], s_38_0, s_39_0)/*19*/
        .invoke(ADD_1, FIFO_C_SHF[40], FIFO_C_SHF[41], s_40_0, s_41_0)/*20*/
        .invoke(ADD_0, FIFO_C_SHF[42], FIFO_C_SHF[43], s_42_0, s_43_0)/*21*/
        .invoke(ADD_1, FIFO_C_SHF[44], FIFO_C_SHF[45], s_44_0, s_45_0)/*22*/
        .invoke(ADD_0, FIFO_C_SHF[46], FIFO_C_SHF[47], s_46_0, s_47_0)/*23*/
        .invoke(ADD_1, FIFO_C_SHF[48], FIFO_C_SHF[49], s_48_0, s_49_0)/*24*/
        .invoke(ADD_0, FIFO_C_SHF[50], FIFO_C_SHF[51], s_50_0, s_51_0)/*25*/
        .invoke(ADD_1, FIFO_C_SHF[52], FIFO_C_SHF[53], s_52_0, s_53_0)/*26*/
        .invoke(ADD_0, FIFO_C_SHF[54], FIFO_C_SHF[55], s_54_0, s_55_0)/*27*/
        .invoke(ADD_1, FIFO_C_SHF[56], FIFO_C_SHF[57], s_56_0, s_57_0)/*28*/
        .invoke(ADD_0, FIFO_C_SHF[58], FIFO_C_SHF[59], s_58_0, s_59_0)/*29*/
        .invoke(ADD_1, FIFO_C_SHF[60], FIFO_C_SHF[61], s_60_0, s_61_0)/*30*/
        .invoke(ADD_0, FIFO_C_SHF[62], FIFO_C_SHF[63], s_62_0, s_63_0)/*31*/
        .invoke(ADD_1, s_1_0, s_2_0, s_1_1, s_2_1)/*32*/
        .invoke(ADD_0, s_5_0, s_6_0, s_5_1, s_6_1)/*33*/
        .invoke(ADD_1, s_9_0, s_10_0, s_9_1, s_10_1)/*34*/
        .invoke(ADD_0, s_13_0, s_14_0, s_13_1, s_14_1)/*35*/
        .invoke(ADD_1, s_17_0, s_18_0, s_17_1, s_18_1)/*36*/
        .invoke(ADD_0, s_21_0, s_22_0, s_21_1, s_22_1)/*37*/
        .invoke(ADD_1, s_25_0, s_26_0, s_25_1, s_26_1)/*38*/
        .invoke(ADD_0, s_29_0, s_30_0, s_29_1, s_30_1)/*39*/
        .invoke(ADD_1, s_33_0, s_34_0, s_33_1, s_34_1)/*40*/
        .invoke(ADD_0, s_37_0, s_38_0, s_37_1, s_38_1)/*41*/
        .invoke(ADD_1, s_41_0, s_42_0, s_41_1, s_42_1)/*42*/
        .invoke(ADD_0, s_45_0, s_46_0, s_45_1, s_46_1)/*43*/
        .invoke(ADD_1, s_49_0, s_50_0, s_49_1, s_50_1)/*44*/
        .invoke(ADD_0, s_53_0, s_54_0, s_53_1, s_54_1)/*45*/
        .invoke(ADD_1, s_57_0, s_58_0, s_57_1, s_58_1)/*46*/
        .invoke(ADD_0, s_61_0, s_62_0, s_61_1, s_62_1)/*47*/
        .invoke(SSW, s_2_1, s_3_0, s_2_2, s_3_1)/*48*/
        .invoke(SSW, s_4_0, s_5_1, s_4_1, s_5_2)/*49*/
        .invoke(ADD_1, s_3_1, s_4_1, s_3_2, s_4_2)/*50*/
        .invoke(SSW, s_10_1, s_11_0, s_10_2, s_11_1)/*51*/
        .invoke(SSW, s_12_0, s_13_1, s_12_1, s_13_2)/*52*/
        .invoke(ADD_0, s_11_1, s_12_1, s_11_2, s_12_2)/*53*/
        .invoke(SSW, s_18_1, s_19_0, s_18_2, s_19_1)/*54*/
        .invoke(SSW, s_20_0, s_21_1, s_20_1, s_21_2)/*55*/
        .invoke(ADD_1, s_19_1, s_20_1, s_19_2, s_20_2)/*56*/
        .invoke(SSW, s_26_1, s_27_0, s_26_2, s_27_1)/*57*/
        .invoke(SSW, s_28_0, s_29_1, s_28_1, s_29_2)/*58*/
        .invoke(ADD_0, s_27_1, s_28_1, s_27_2, s_28_2)/*59*/
        .invoke(SSW, s_34_1, s_35_0, s_34_2, s_35_1)/*60*/
        .invoke(SSW, s_36_0, s_37_1, s_36_1, s_37_2)/*61*/
        .invoke(ADD_1, s_35_1, s_36_1, s_35_2, s_36_2)/*62*/
        .invoke(SSW, s_42_1, s_43_0, s_42_2, s_43_1)/*63*/
        .invoke(SSW, s_44_0, s_45_1, s_44_1, s_45_2)/*64*/
        .invoke(ADD_0, s_43_1, s_44_1, s_43_2, s_44_2)/*65*/
        .invoke(SSW, s_50_1, s_51_0, s_50_2, s_51_1)/*66*/
        .invoke(SSW, s_52_0, s_53_1, s_52_1, s_53_2)/*67*/
        .invoke(ADD_1, s_51_1, s_52_1, s_51_2, s_52_2)/*68*/
        .invoke(SSW, s_58_1, s_59_0, s_58_2, s_59_1)/*69*/
        .invoke(SSW, s_60_0, s_61_1, s_60_1, s_61_2)/*70*/
        .invoke(ADD_0, s_59_1, s_60_1, s_59_2, s_60_2)/*71*/
        .invoke(SSW, s_4_2, s_7_0, s_4_3, s_7_1)/*72*/
        .invoke(SSW, s_8_0, s_11_2, s_8_1, s_11_3)/*73*/
        .invoke(ADD_1, s_7_1, s_8_1, s_7_2, s_8_2)/*74*/
        .invoke(SSW, s_20_2, s_23_0, s_20_3, s_23_1)/*75*/
        .invoke(SSW, s_24_0, s_27_2, s_24_1, s_27_3)/*76*/
        .invoke(ADD_0, s_23_1, s_24_1, s_23_2, s_24_2)/*77*/
        .invoke(SSW, s_36_2, s_39_0, s_36_3, s_39_1)/*78*/
        .invoke(SSW, s_40_0, s_43_2, s_40_1, s_43_3)/*79*/
        .invoke(ADD_1, s_39_1, s_40_1, s_39_2, s_40_2)/*80*/
        .invoke(SSW, s_52_2, s_55_0, s_52_3, s_55_1)/*81*/
        .invoke(SSW, s_56_0, s_59_2, s_56_1, s_59_3)/*82*/
        .invoke(ADD_0, s_55_1, s_56_1, s_55_2, s_56_2)/*83*/
        .invoke(SSW, s_8_2, s_15_0, s_8_3, s_15_1)/*84*/
        .invoke(SSW, s_16_0, s_23_2, s_16_1, s_23_3)/*85*/
        .invoke(ADD_1, s_15_1, s_16_1, s_15_2, s_16_2)/*86*/
        .invoke(SSW, s_40_2, s_47_0, s_40_3, s_47_1)/*87*/
        .invoke(SSW, s_48_0, s_55_2, s_48_1, s_55_3)/*88*/
        .invoke(ADD_0, s_47_1, s_48_1, s_47_2, s_48_2)/*89*/
        .invoke(SSW, s_16_2, s_31_0, s_16_3, s_31_1)/*90*/
        .invoke(SSW, s_32_0, s_47_2, s_32_1, s_47_3)/*91*/
        .invoke(ADD_X, s_31_1, s_32_1, s_31_2, s_32_2)/*92*/
        .invoke(SSW, s_16_3, s_31_2, s_16_4, s_31_3)/*93*/
        .invoke(SSW, s_32_2, s_47_3, s_32_3, s_47_4)/*94*/
        .invoke(SWB1_4, s_15_2, s_16_4, s_15_3, s_16_5)/*95*/
        .invoke(SSW, s_8_3, s_15_3, s_8_4, s_15_4)/*96*/
        .invoke(SSW, s_16_5, s_23_3, s_16_6, s_23_4)/*97*/
        .invoke(SWB0_4, s_47_4, s_48_2, s_47_5, s_48_3)/*98*/
        .invoke(SSW, s_40_3, s_47_5, s_40_4, s_47_6)/*99*/
        .invoke(SSW, s_48_3, s_55_3, s_48_4, s_55_4)/*100*/
        .invoke(SWB1_3, s_7_2, s_8_4, s_7_3, s_8_5)/*101*/
        .invoke(SSW, s_4_3, s_7_3, s_4_4, s_7_4)/*102*/
        .invoke(SSW, s_8_5, s_11_3, s_8_6, s_11_4)/*103*/
        .invoke(SWB0_3, s_23_4, s_24_2, s_23_5, s_24_3)/*104*/
        .invoke(SSW, s_20_3, s_23_5, s_20_4, s_23_6)/*105*/
        .invoke(SSW, s_24_3, s_27_3, s_24_4, s_27_4)/*106*/
        .invoke(SWB1_3, s_39_2, s_40_4, s_39_3, s_40_5)/*107*/
        .invoke(SSW, s_36_3, s_39_3, s_36_4, s_39_4)/*108*/
        .invoke(SSW, s_40_5, s_43_3, s_40_6, s_43_4)/*109*/
        .invoke(SWB0_3, s_55_4, s_56_2, s_55_5, s_56_3)/*110*/
        .invoke(SSW, s_52_3, s_55_5, s_52_4, s_55_6)/*111*/
        .invoke(SSW, s_56_3, s_59_3, s_56_4, s_59_4)/*112*/
        .invoke(SWB1_2, s_3_2, s_4_4, s_3_3, s_4_5)/*113*/
        .invoke(SSW, s_2_2, s_3_3, s_2_3, s_3_4)/*114*/
        .invoke(SSW, s_4_5, s_5_2, s_4_6, s_5_3)/*115*/
        .invoke(SWB0_2, s_11_4, s_12_2, s_11_5, s_12_3)/*116*/
        .invoke(SSW, s_10_2, s_11_5, s_10_3, s_11_6)/*117*/
        .invoke(SSW, s_12_3, s_13_2, s_12_4, s_13_3)/*118*/
        .invoke(SWB1_2, s_19_2, s_20_4, s_19_3, s_20_5)/*119*/
        .invoke(SSW, s_18_2, s_19_3, s_18_3, s_19_4)/*120*/
        .invoke(SSW, s_20_5, s_21_2, s_20_6, s_21_3)/*121*/
        .invoke(SWB0_2, s_27_4, s_28_2, s_27_5, s_28_3)/*122*/
        .invoke(SSW, s_26_2, s_27_5, s_26_3, s_27_6)/*123*/
        .invoke(SSW, s_28_3, s_29_2, s_28_4, s_29_3)/*124*/
        .invoke(SWB1_2, s_35_2, s_36_4, s_35_3, s_36_5)/*125*/
        .invoke(SSW, s_34_2, s_35_3, s_34_3, s_35_4)/*126*/
        .invoke(SSW, s_36_5, s_37_2, s_36_6, s_37_3)/*127*/
        .invoke(SWB0_2, s_43_4, s_44_2, s_43_5, s_44_3)/*128*/
        .invoke(SSW, s_42_2, s_43_5, s_42_3, s_43_6)/*129*/
        .invoke(SSW, s_44_3, s_45_2, s_44_4, s_45_3)/*130*/
        .invoke(SWB1_2, s_51_2, s_52_4, s_51_3, s_52_5)/*131*/
        .invoke(SSW, s_50_2, s_51_3, s_50_3, s_51_4)/*132*/
        .invoke(SSW, s_52_5, s_53_2, s_52_6, s_53_3)/*133*/
        .invoke(SWB0_2, s_59_4, s_60_2, s_59_5, s_60_3)/*134*/
        .invoke(SSW, s_58_2, s_59_5, s_58_3, s_59_6)/*135*/
        .invoke(SSW, s_60_3, s_61_2, s_60_4, s_61_3)/*136*/
        .invoke(SWB1_1, s_1_1, s_2_3, s_1_2, s_2_4)/*137*/
        .invoke(SWB0_1, s_5_3, s_6_1, s_5_4, s_6_2)/*138*/
        .invoke(SWB1_1, s_9_1, s_10_3, s_9_2, s_10_4)/*139*/
        .invoke(SWB0_1, s_13_3, s_14_1, s_13_4, s_14_2)/*140*/
        .invoke(SWB1_1, s_17_1, s_18_3, s_17_2, s_18_4)/*141*/
        .invoke(SWB0_1, s_21_3, s_22_1, s_21_4, s_22_2)/*142*/
        .invoke(SWB1_1, s_25_1, s_26_3, s_25_2, s_26_4)/*143*/
        .invoke(SWB0_1, s_29_3, s_30_1, s_29_4, s_30_2)/*144*/
        .invoke(SWB1_1, s_33_1, s_34_3, s_33_2, s_34_4)/*145*/
        .invoke(SWB0_1, s_37_3, s_38_1, s_37_4, s_38_2)/*146*/
        .invoke(SWB1_1, s_41_1, s_42_3, s_41_2, s_42_4)/*147*/
        .invoke(SWB0_1, s_45_3, s_46_1, s_45_4, s_46_2)/*148*/
        .invoke(SWB1_1, s_49_1, s_50_3, s_49_2, s_50_4)/*149*/
        .invoke(SWB0_1, s_53_3, s_54_1, s_53_4, s_54_2)/*150*/
        .invoke(SWB1_1, s_57_1, s_58_3, s_57_2, s_58_4)/*151*/
        .invoke(SWB0_1, s_61_3, s_62_1, s_61_4, s_62_2)/*152*/
        .invoke(SWB1_0, s_0_0, s_1_2, FIFO_C_BUF[0], FIFO_C_BUF[1])/*153*/
        .invoke(SWB0_0, s_2_4, s_3_4, FIFO_C_BUF[2], FIFO_C_BUF[3])/*154*/
        .invoke(SWB1_0, s_4_6, s_5_4, FIFO_C_BUF[4], FIFO_C_BUF[5])/*155*/
        .invoke(SWB0_0, s_6_2, s_7_4, FIFO_C_BUF[6], FIFO_C_BUF[7])/*156*/
        .invoke(SWB1_0, s_8_6, s_9_2, FIFO_C_BUF[8], FIFO_C_BUF[9])/*157*/
        .invoke(SWB0_0, s_10_4, s_11_6, FIFO_C_BUF[10], FIFO_C_BUF[11])/*158*/
        .invoke(SWB1_0, s_12_4, s_13_4, FIFO_C_BUF[12], FIFO_C_BUF[13])/*159*/
        .invoke(SWB0_0, s_14_2, s_15_4, FIFO_C_BUF[14], FIFO_C_BUF[15])/*160*/
        .invoke(SWB1_0, s_16_6, s_17_2, FIFO_C_BUF[16], FIFO_C_BUF[17])/*161*/
        .invoke(SWB0_0, s_18_4, s_19_4, FIFO_C_BUF[18], FIFO_C_BUF[19])/*162*/
        .invoke(SWB1_0, s_20_6, s_21_4, FIFO_C_BUF[20], FIFO_C_BUF[21])/*163*/
        .invoke(SWB0_0, s_22_2, s_23_6, FIFO_C_BUF[22], FIFO_C_BUF[23])/*164*/
        .invoke(SWB1_0, s_24_4, s_25_2, FIFO_C_BUF[24], FIFO_C_BUF[25])/*165*/
        .invoke(SWB0_0, s_26_4, s_27_6, FIFO_C_BUF[26], FIFO_C_BUF[27])/*166*/
        .invoke(SWB1_0, s_28_4, s_29_4, FIFO_C_BUF[28], FIFO_C_BUF[29])/*167*/
        .invoke(SWB0_0, s_30_2, s_31_3, FIFO_C_BUF[30], FIFO_C_BUF[31])/*168*/
        .invoke(SWB1_0, s_32_3, s_33_2, FIFO_C_BUF[32], FIFO_C_BUF[33])/*169*/
        .invoke(SWB0_0, s_34_4, s_35_4, FIFO_C_BUF[34], FIFO_C_BUF[35])/*170*/
        .invoke(SWB1_0, s_36_6, s_37_4, FIFO_C_BUF[36], FIFO_C_BUF[37])/*171*/
        .invoke(SWB0_0, s_38_2, s_39_4, FIFO_C_BUF[38], FIFO_C_BUF[39])/*172*/
        .invoke(SWB1_0, s_40_6, s_41_2, FIFO_C_BUF[40], FIFO_C_BUF[41])/*173*/
        .invoke(SWB0_0, s_42_4, s_43_6, FIFO_C_BUF[42], FIFO_C_BUF[43])/*174*/
        .invoke(SWB1_0, s_44_4, s_45_4, FIFO_C_BUF[44], FIFO_C_BUF[45])/*175*/
        .invoke(SWB0_0, s_46_2, s_47_6, FIFO_C_BUF[46], FIFO_C_BUF[47])/*176*/
        .invoke(SWB1_0, s_48_4, s_49_2, FIFO_C_BUF[48], FIFO_C_BUF[49])/*177*/
        .invoke(SWB0_0, s_50_4, s_51_4, FIFO_C_BUF[50], FIFO_C_BUF[51])/*178*/
        .invoke(SWB1_0, s_52_6, s_53_4, FIFO_C_BUF[52], FIFO_C_BUF[53])/*179*/
        .invoke(SWB0_0, s_54_2, s_55_6, FIFO_C_BUF[54], FIFO_C_BUF[55])/*180*/
        .invoke(SWB1_0, s_56_4, s_57_2, FIFO_C_BUF[56], FIFO_C_BUF[57])/*181*/
        .invoke(SWB0_0, s_58_4, s_59_6, FIFO_C_BUF[58], FIFO_C_BUF[59])/*182*/
        .invoke(SWB1_0, s_60_4, s_61_4, FIFO_C_BUF[60], FIFO_C_BUF[61])/*183*/
        .invoke(SWB0_0, s_62_2, s_63_0, FIFO_C_BUF[62], FIFO_C_BUF[63])/*184*/
        .invoke<tapa::join, NUM_PES>(Accumulator, FIFO_C_ARB, FIFO_C_BUF, M, numTilesM, last_tile_idx)
        .invoke<tapa::detach, 4>(Arbiter_C_16_1, FIFO_C_ARB, FIFO_C_AB)
        .invoke<tapa::detach, NUM_C_CH>(Compute_C, FIFO_C_AB, FIFO_C_IN, FIFO_C_OUT, alpha, beta)
        .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, C_out, numTilesM, numTilesN, M, N, rp_time)
        ;
}
