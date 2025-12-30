
void MM2S_B(tapa::async_mmap<float_vB>& mmap,
    tapa::ostream<float_vB>& stream,
    const uint32_t numTilesM, const uint32_t numTilesN, const uint32_t numTilesK,
    const uint32_t K, const uint16_t rp_time) {
    
  for (int r = 0; r < rp_time; r++) {
    for (int n = 0; n < numTilesN; n++) {
      for (int m = 0; m < numTilesM; m++) {
        for (int k = 0; k < numTilesK; k++) {
          #pragma HLS loop_flatten OFF
          int read_len = ((k == (numTilesK - 1)) && (K%K0 != 0)) ? ((K%K0)/KX) : B_READ_LEN; //last tile might not be having B_REAd_LEN adresses
          int start_addr = (k * numTilesN * B_READ_LEN) + (n * read_len); // each tile in a channel takes B_READ_LEN addresses except last k tiles
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
  }
  printf("MM2S_B done\n");
}
