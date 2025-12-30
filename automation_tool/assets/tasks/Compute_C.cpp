
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
