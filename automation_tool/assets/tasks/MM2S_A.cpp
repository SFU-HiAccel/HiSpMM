
void MM2S_A(tapa::async_mmap<uint64_v>& mmap,
    tapa::ostreams<uint64_v2, PES_PER_CH/2>& streams,
    const uint32_t len, const uint16_t numTilesN, const uint16_t rp_time) {
      printf("MM2S_A\n");
    for(uint32_t rp = 0; rp < rp_time * numTilesN; rp++) {
        for(uint32_t i_req = 0, i_resp = 0; i_resp < len;) {
        #pragma HLS pipeline II=1
        async_readA(mmap,
                    streams,
                    len,
                    i_req, i_resp); 
         
        }
    }
    printf("MM2S_A done\n");
}
