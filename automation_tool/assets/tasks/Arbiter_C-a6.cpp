
void Arbiter_C(tapa::istreams<float_vN, NUM_PES>& c_arb,
    tapa::ostreams<float_vB, NUM_C_CH>& c_ab) {


  float_vB temp_out[NUM_C_CH];
  float_vN temp_in;
  
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
  }
}