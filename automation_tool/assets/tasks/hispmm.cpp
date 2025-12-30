void hispmm(tapa::mmaps<uint64_v, NUM_A_CH> A,
    tapa::mmaps<float_vB, NUM_B_CH> B_in,
    tapa::mmaps<float_vB, NUM_C_CH> C_in,
    tapa::mmaps<float_vB, NUM_C_CH> C_out,
    const float alpha, const float beta,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t numTilesM, const uint32_t numTilesN, const uint32_t numTilesK,
    const uint32_t len, const uint32_t last_tile_idx, const uint16_t rp_time)
{
tapa::streams<uint64_v2, NUM_PES_HALF, FIFO_DEPTH> FIFO_A_IN("a_in");
tapa::streams<float_vB, ((NUM_PEG + 1)* NUM_B_CH), FIFO_DEPTH> FIFO_B_IN("b_in");
tapa::streams<float_vN, NUM_PES, FIFO_LARGE_DEPTH> FIFO_C_ARB("c_arb");
tapa::streams<float_vB, NUM_C_CH, FIFO_LARGE_DEPTH> FIFO_C_AB("c_ab");
tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_IN("c_in");
tapa::streams<float_vB, NUM_C_CH, FIFO_DEPTH> FIFO_C_OUT("c_out");
#ifdef RS_DESIGN
tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
#else
tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
#endif
#if NUM_PES == 80
tapa::streams<float_vN, 8, FIFO_LARGE_DEPTH> FIFO_C_AB_INTER("c_ab_inter");
#endif

tapa::task()
  .invoke<tapa::join, NUM_A_CH>(MM2S_A, A, FIFO_A_IN, len, numTilesN, rp_time)
  .invoke<tapa::join, NUM_B_CH>(MM2S_B, B_in, FIFO_B_IN, numTilesM, numTilesN, numTilesK, K, rp_time)
  .invoke<tapa::join, NUM_C_CH>(MM2S_C, C_in, FIFO_C_IN, numTilesM, numTilesN, M, N, rp_time)
  .invoke<tapa::join, NUM_PEG>(PEG, FIFO_A_IN, FIFO_B_IN, FIFO_B_IN, FIFO_C_SHF, K, numTilesK, last_tile_idx)
  .invoke<tapa::detach, NUM_B_CH>(DummyRead, FIFO_B_IN)
  .invoke<tapa::join, NUM_PES>(Accumulator, FIFO_C_ARB, FIFO_C_SHF, M, numTilesM, last_tile_idx)
#if NUM_PES == 80
  .invoke<tapa::detach, 8>(Arbiter_C_10_1,  FIFO_C_ARB, FIFO_C_AB_INTER)
  .invoke<tapa::detach>(Arbiter_C_8_4,  FIFO_C_AB_INTER, FIFO_C_AB)
#elif (NUM_PES == 64) && (NUM_C_CH == 8)
  .invoke<tapa::detach, 8>(Arbiter_C_8_1,  FIFO_C_ARB, FIFO_C_AB)
#elif (NUM_PES == 64) && (NUM_C_CH == 4)
  .invoke<tapa::detach, 4>(Arbiter_C_16_1,  FIFO_C_ARB, FIFO_C_AB)
#else
  .invoke<tapa::detach>(Arbiter_C,  FIFO_C_ARB, FIFO_C_AB)
#endif
  .invoke<tapa::detach, NUM_C_CH>(Compute_C,  FIFO_C_AB, FIFO_C_IN, FIFO_C_OUT, alpha, beta)
  .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, C_out, numTilesM, numTilesN, M, N, rp_time)
  ;
}