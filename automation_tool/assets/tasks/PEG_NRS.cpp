
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
  