
.PHONY: host tapa hw-build hw-test sw-test clean

CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wno-unused-result -Wno-write-strings
INCLUDES = -I${XILINX_HLS}/include
LDFLAGS = -ltapa -lfrt -lglog -lgflags -lOpenCL
HOST_FLAGS = --verbose

SRCS = src/hispmm.cpp src/hispmm_host.cpp src/prepare_amt_unified.cpp
TARGET = hispmm
platform = xilinx_u280_gen3x16_xdma_1_202211_1

# Defaults (so this file can be included without extra config fragments).
CONNECTIVITY_FILE ?= link_config.ini
FLOORPLAN_PREASSIGNMENTS ?= floorplan.json
FLOORPLAN_STRATEGY ?= HALF_SLR_LEVEL_FLOORPLANNING
MAX_AREA_LIMIT ?= 0.95
CLOCK_PERIOD ?= 4.255

host: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS)

tapa:
	tapac -o hispmm.$(platform).hw.xo src/hispmm.cpp \
  --connectivity $(CONNECTIVITY_FILE) \
  --platform $(platform) \
  --top hispmm \
  --read-only-args "A*" \
  --read-only-args "B_in*" \
  --read-only-args "C_in*" \
  --write-only-args "C_out*" \
  --work-dir hispmm.$(platform).hw.xo.tapa \
  --enable-hbm-binding-adjustment \
  --enable-floorplan \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --floorplan-output constraint.tcl \
  --floorplan-strategy $(FLOORPLAN_STRATEGY) \
  --floorplan-pre-assignments $(FLOORPLAN_PREASSIGNMENTS) \
  --max-area-limit $(MAX_AREA_LIMIT) \
  --clock-period $(CLOCK_PERIOD)

hw-build:
	sed -i 's/STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.OPT_DESIGN.ARGS.DIRECTIVE="ExploreWithRemap"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$$PLACEMENT_STRATEGY/STEPS.PLACE_DESIGN.ARGS.DIRECTIVE="ExtraPostPlacementOpt"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE="AggressiveExplore"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sed -i 's/STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$$STRATEGY/STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE="MoreGlobalIterations"/g' hispmm.$(platform).hw_generate_bitstream.sh
	sh hispmm.$(platform).hw_generate_bitstream.sh

sw-test:
	./hispmm $(HOST_FLAGS) matrices/hangGlider_3.mtx 1 32

hw-test:
	./hispmm $(HOST_FLAGS) --bitstream=vitis_run_hw/hispmm.$(platform).hw.xclbin matrices/hangGlider_3.mtx 1000

clean:
	rm -f $(TARGET)


