
void Arbiter_C_8_4( tapa::istreams<float_vN, 8>& c_arb,
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
