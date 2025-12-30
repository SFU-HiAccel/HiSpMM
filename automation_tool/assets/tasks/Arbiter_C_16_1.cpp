
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
