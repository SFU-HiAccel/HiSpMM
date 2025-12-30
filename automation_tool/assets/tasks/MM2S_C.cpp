
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
