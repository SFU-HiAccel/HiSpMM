.PHONY: host tapa hw-build hw-test sw-test clean

CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wno-unused-result -Wno-write-strings
INCLUDES = -I${XILINX_HLS}/include
LDFLAGS = -ltapa -lfrt -lglog -lgflags -lOpenCL

SRCS = src/spmm.cpp src/spmm-host.cpp src/helper_functions.cpp
TARGET = hispmm
platform = xilinx_u280_gen3x16_xdma_1_202211_1


host: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS)

tapa:
	tapac -o hispmm.$(platform).hw.xo src/spmm.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top hispmm \
  --read-only-args "A*" \
  --read-only-args "B_in*" \
  --read-only-args "C_in*" \
  --write-only-args "C_out*" \
  --work-dir hispmm.$(platform).hw.xo.tapa \
  --enable-hbm-binding-adjustment \
  --enable-floorplan \
  --enable-buffer-support \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --floorplan-strategy SLR_LEVEL_FLOORPLANNING \
  --max-area-limit 0.95 \
  --floorplan-pre-assignments floorplan_slr.json \
  --floorplan-output constraint.tcl \
  --clock-period 4.255

hw-build:
	sed -i 's/STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.OPT_DESIGN.ARGS.DIRECTIVE="ExploreWithRemap"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$$PLACEMENT_STRATEGY/STEPS.PLACE_DESIGN.ARGS.DIRECTIVE="ExtraPostPlacementOpt"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE="AggressiveExplore"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE="MoreGlobalIterations"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sh hispmm.$(platform).hw_generate_bitstream.sh

sw-test:
	./hispmm matrices/airfoil_2d.mtx 1 32

hw-test:
	./hispmm --bitstream=vitis_run_hw/hispmm.$(platform).hw.xclbin matrices/airfoil_2d.mtx 1000

clean:
	rm -f $(TARGET)