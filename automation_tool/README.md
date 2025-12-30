### Automation tool (codegen + DSE)

This directory is a **fresh** kernel code generator that:

- **Only consumes task templates** from `assets/tasks/*` for code emission
- Uses existing **cycle/resource analysis** as an advisor (via `automation_tool/dse/cycle_analysis.py`)
- Uses existing **DRDN topology/depth policy** (via `crossbar.py`)

It emits **kernel + host** sources:

- `src/hispmm.h`
- `src/hispmm.cpp`
- `src/hispmm_host.cpp`

### How it works

- **Advisor step**: runs `automation_tool/dse/cycle_analysis.py` and parses the `RECOMMEND:` line (or uses `--pick`).
- **Header step**: patches `assets/tasks/hispmm.h` defines (`NUM_A_CH`, `NUM_C_CH`, `NUM_PES`, etc.).
- **Kernel step**: concatenates task implementations from `assets/tasks/*.cpp` plus a patched top function from `assets/tasks/hispmm.cpp`.
- **Terminology**: **balanced kernel = NRS-only**, **imbalanced kernel = RS-capable**.
- **Imbalanced kernels (RS-capable)**: inject DRDN stream declarations and `.invoke(...)` blocks based on `crossbar.py`’s generated graph.

### Usage

Generate the recommended kernel for a matrix:

```bash
python -m automation_tool --matrix automation_tool/assets/common/matrices/airfoil_2d.mtx --n 8 --out generated/kernel_auto
```

Force a specific label (useful for testing):

```bash
python -m automation_tool --matrix automation_tool/assets/common/matrices/airfoil_2d.mtx --n 8 --out generated/kernel_pick --pick imbalanced_a8_c4
```

### Outputs

- `generated/kernel_auto/src/hispmm.h`
- `generated/kernel_auto/src/hispmm.cpp`
- `generated/kernel_auto/src/hispmm_host.cpp`

### Notes / scope

- This generator **does** emit `Makefile`, `link_config.ini`, and (optionally) `floorplan.json`.
- It’s template-driven via `assets/tasks`.


