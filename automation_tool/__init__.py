"""HiSpMM kernel generator and DSE helper.

This package intentionally treats existing analysis code as a black box:
- Uses `automation_tool/dse/cycle_analysis.py` for performance/resource recommendation (via subprocess).
- Uses `crossbar.py` for DRDN (row sharing network) topology/depth policy (via import).

"""

from __future__ import annotations


