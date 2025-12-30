
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
