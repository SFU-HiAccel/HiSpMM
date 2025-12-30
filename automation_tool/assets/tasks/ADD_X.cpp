
void ADD_X(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    float_vN temp[2];
    bool dummy[2];
    float_vN sum;

    for(bool last = false; !last; ) {
        #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        for (int p = 0; p < N0; p++) // ---> Cnoc_pkt.val is a float val[N0]
          sum[p] = curr_in0.val[p] + curr_in1.val[p];
        
        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy)) // ---> Cnoc_pkt.sharedRow is a bool
        {
          bool i = ((curr_in0.bank >> (LOG_2_NUM_PES - 1)) & 1);
          for (int p = 0; p < N0; p++){ // ---> Cnoc_pkt.val is a float val[N0]
            temp[i][p] = sum[p];
            temp[!i][p] = (float)0;
          }

          dummy[i] = false;
          dummy[!i] = true;
        }

        else {
          for (int p = 0; p < N0; p++) { // ---> Cnoc_pkt.val is a float val[N0]
            temp[0][p] = curr_in0.val[p];
            temp[1][p] = curr_in1.val[p];
          }

          dummy[0] = curr_in0.dummy;
          dummy[1] = curr_in1.dummy;
        }
        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        for (int p = 0; p < N0; p++) { // ---> Cnoc_pkt.val is a float val[N0]
          curr_out0.val[p] = temp[0][p];
          curr_out1.val[p] = temp[1][p];
        }
        c_out0 << curr_out0;
        c_out1 << curr_out1;

        last = curr_in0.last & curr_in1.last;
    }  
}
