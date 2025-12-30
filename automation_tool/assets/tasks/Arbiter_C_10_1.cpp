
void Arbiter_C_10_1( tapa::istreams<float_vN, 10>& c_arb,
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

