

     
void DummyRead(tapa::istream<float_vB>& b_in) {
  for (;;) {
  #pragma HLS PIPELINE II=1
    float_vB temp = b_in.read();
  }
}
