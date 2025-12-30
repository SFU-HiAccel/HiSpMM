
void S2MM_C(tapa::istream<float_vB>& stream,
    tapa::async_mmap<float_vB>& mmap,
    const uint32_t numTilesM, const uint32_t numTilesN, 
    const uint32_t M, const uint32_t N, const uint16_t rp_time) {
    for (int r = 0; r < rp_time; r++) {
      for (int n = 0; n < numTilesN; n++) {
        for (int m = 0; m < numTilesM; m++) {     
          #pragma HLS loop_flatten OFF         
          int read_len = ((m == (numTilesM - 1)) && (M%M0 != 0)) ? ((M%M0)/MX) : C_READ_LEN; //last tile might not be having C_REAd_LEN adresses
          int start_addr = (m * numTilesN * C_READ_LEN) + (n * read_len); // each tile in a channel takes C_READ_LEN addresses except last k tiles
          async_writeC(mmap, stream, start_addr, read_len);
        }
      }
    }
    printf("S2MM_C done\n");
}
