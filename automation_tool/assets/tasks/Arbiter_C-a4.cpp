
void Arbiter_C(tapa::istreams<float_vN, NUM_PES>& c_arb,
    tapa::ostreams<float_vB, NUM_C_CH>& c_ab) {
 
  for (;;) {
    // NOTE: For A4, NUM_PES is compile-time constant (NUM_PES = A_CH * 8 = 32).
    // The (p += NUM_C_CH*2) loop is logically correct, but some HLS flows may not
    // synthesize it as intended. We "open" (unroll) the loop into explicit phases.
    //
    // Each phase reads (NUM_C_CH*2) float_vN packets and writes NUM_C_CH float_vB packets.

#if NUM_C_CH == 8
    // Phase 0: p = 0, reads c_arb[0..15] -> writes c_ab[0..7]
    #pragma HLS PIPELINE II=1
    float_vB temp_out0[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[0 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out0[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out0[c];

    // Phase 1: p = 16, reads c_arb[16..31] -> writes c_ab[0..7]
    #pragma HLS PIPELINE II=1
    float_vB temp_out1[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[16 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out1[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out1[c];

#elif NUM_C_CH == 4
    // For C4: each phase reads 8 inputs and writes 4 outputs; 4 phases cover 32 inputs.
    // Phase 0: p = 0, reads c_arb[0..7]
    #pragma HLS PIPELINE II=1
    float_vB temp_out0[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[0 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out0[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out0[c];

    // Phase 1: p = 8, reads c_arb[8..15]
    #pragma HLS PIPELINE II=1
    float_vB temp_out1[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[8 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out1[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out1[c];

    // Phase 2: p = 16, reads c_arb[16..23]
    #pragma HLS PIPELINE II=1
    float_vB temp_out2[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[16 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out2[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out2[c];

    // Phase 3: p = 24, reads c_arb[24..31]
    #pragma HLS PIPELINE II=1
    float_vB temp_out3[NUM_C_CH];
    for (int j = 0; j < NUM_C_CH * 2; j++) {
      float_vN temp_in = c_arb[24 + j].read();
      for (int k = 0; k < N0; k++) {
        temp_out3[j / 2][(j % 2) * N0 + k] = temp_in[k];
      }
    }
    for (int c = 0; c < NUM_C_CH; c++) c_ab[c] << temp_out3[c];

#else
#error "Arbiter_C-a4.cpp: Unsupported NUM_C_CH for A4 monolithic arbiter. Expected NUM_C_CH==4 or NUM_C_CH==8."
#endif
  }

}