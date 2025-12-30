
void SWB0_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cvec_pkt>& c_out0, tapa::ostream<Cvec_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cvec_pkt curr_out[2];

        bool i = (curr_in0.bank & 1) && curr_in0.sharedRow;
        curr_out[i].dummy = curr_in0.dummy;
        curr_out[!i].dummy = curr_in1.dummy;
        curr_out[i].tileEnd = curr_in0.tileEnd;
        curr_out[!i].tileEnd = curr_in1.tileEnd;
        curr_out[i].row16 = curr_in0.row16;
        curr_out[!i].row16 = curr_in1.row16;
        for (int p = 0; p < N0; p++) {
          curr_out[i].val[p] = curr_in0.val[p];
          curr_out[!i].val[p] = curr_in1.val[p];
        }

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}
