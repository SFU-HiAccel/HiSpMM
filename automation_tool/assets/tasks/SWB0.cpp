
template<int n>
void SWB0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];


        bool i = ((curr_in0.bank >> n) & 1) && (curr_in0.sharedRow);
        curr_out[i] = curr_in0;
        curr_out[!i] = curr_in1;
        
        c_out0 << curr_out[0];
        c_out1 << curr_out[1];
        last = curr_in0.last & curr_in1.last;    
    }
}
