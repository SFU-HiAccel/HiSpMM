from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Literal, TypeAlias


ResourceName: TypeAlias = Literal["BRAM", "DSP", "FF", "LUT", "URAM"]

BUILTIN_CAPACITY: dict[ResourceName, float] = {
    "BRAM": 3504.0,
    "DSP": 8496.0,
    "FF": 2331840.0,
    "LUT": 1165920.0,
    "URAM": 960.0,
}


def builtin_area_model() -> tuple[dict[ResourceName, float], list["TaskArea"]]:
    """
    Built-in resource model snapshot (embedded from `area.log`) so the tool does not
    depend on an external file at runtime.
    """
    tasks: list[TaskArea] = [
        TaskArea("ADD_0", "TBD", 0.0, 16.0, 3063.0, 3400.625, 0.0),
        TaskArea("ADD_1", "TBD", 0.0, 16.0, 3063.0, 3400.625, 0.0),
        TaskArea("ADD_X", "TBD", 0.0, 16.0, 3063.0, 3400.625, 0.0),
        TaskArea("Accumulator", "NUM_PES", 0.0, 16.0, 2935.0, 2944.1875, 8.0),
        # Arbiter variants
        TaskArea("Arbiter_C", "4 if NUM_PES == 64", 0.0, 0.0, 3764.0, 6777.0, 0.0),
        TaskArea("Arbiter_C", "1 if NUM_PES == 48", 0.0, 0.0, 8713.0, 18158.0, 0.0),
        TaskArea("Arbiter_C", "1 if NUM_PES == 32", 0.0, 0.0, 2383.0, 13448.0, 0.0),
        TaskArea("Arbiter_C_10_1", "8 if NUM_PES == 80", 0.0, 0.0, 2481.0, 5609.25, 0.0),
        TaskArea("Arbiter_C_8_4_0", "1 if NUM_PES == 80", 0.0, 0.0, 134.0, 3542.0, 0.0),
        TaskArea("Compute_C", "NUM_C_CH", 0.0, 128.0, 9500.0, 7812.125, 0.0),
        TaskArea("DummyRead", "NUM_B_CH", 0.0, 0.0, 16.0, 569.0625, 0.0),
        TaskArea("MM2S_A", "NUM_A_CH", 0.0, 1.0, 103.0, 64.0, 0.0),
        TaskArea("MM2S_B", "NUM_B_CH", 0.0, 2.0, 532.0, 234.0, 0.0),
        TaskArea("MM2S_C", "NUM_C_CH", 0.0, 2.0, 481.0, 218.0, 0.0),
        TaskArea("PEG", "NUM_PES", 128.0, 96.0, 5279.0, 8324.375, 0.0),
        TaskArea("S2MM_C", "NUM_C_CH", 0.0, 2.0, 491.0, 789.0625, 0.0),
        # Crossbar network blocks (RS-only; counts come from crossbar.py)
        TaskArea("SSW", "TBD", 0.0, 0.0, 609.0, 1209.0, 0.0),
        TaskArea("SWB0_0", "TBD", 0.0, 0.0, 579.5, 1004.375, 0.0),
        TaskArea("SWB0_1", "TBD", 0.0, 0.0, 612.0, 2138.25, 0.0),
        TaskArea("SWB0_2", "TBD", 0.0, 0.0, 609.0, 1885.875, 0.0),
        TaskArea("SWB0_3", "TBD", 0.0, 0.0, 609.0, 1636.5, 0.0),
        TaskArea("SWB0_4", "TBD", 0.0, 0.0, 606.0, 1277.25, 0.0),
        TaskArea("SWB1_0", "TBD", 0.0, 0.0, 590.0, 2190.5, 0.0),
        TaskArea("SWB1_1", "TBD", 0.0, 0.0, 612.0, 2137.25, 0.0),
        TaskArea("SWB1_2", "TBD", 0.0, 0.0, 609.0, 1884.875, 0.0),
        TaskArea("SWB1_3", "TBD", 0.0, 0.0, 609.0, 1635.5, 0.0),
        TaskArea("SWB1_4", "TBD", 0.0, 0.0, 606.0, 1276.25, 0.0),
        # AXI + controllers
        TaskArea("A_0__m_axi", "NUM_A_CH", 0.0, 0.0, 369.0, 1735.0, 0.0),
        TaskArea("B_in_0__m_axi", "NUM_B_CH", 0.0, 0.0, 369.0, 1735.0, 0.0),
        TaskArea("C_in_0__m_axi", "NUM_C_CH", 0.0, 0.0, 369.0, 1735.0, 0.0),
        TaskArea("C_out_0__m_axi", "NUM_C_CH", 0.0, 0.0, 369.0, 1735.0, 0.0),
        TaskArea("B_in_0_external_controller", "NUM_B_CH", 0.0, 0.0, 6500.0, 5000.0, 0.0),
        TaskArea("A_0_external_controller", "NUM_A_CH", 0.0, 0.0, 6500.0, 5000.0, 0.0),
        TaskArea("C_in_0_external_controller", "NUM_C_CH", 0.0, 0.0, 6500.0, 5000.0, 0.0),
        TaskArea("C_out_0_external_controller", "NUM_C_CH", 0.0, 0.0, 6500.0, 5000.0, 0.0),
    ]
    return dict(BUILTIN_CAPACITY), tasks


def load_area_model(area_log_path: str | Path | None) -> tuple[dict[ResourceName, float], list["TaskArea"]]:
    """Load resource model from file if provided; otherwise use the built-in snapshot."""
    if area_log_path:
        return parse_area_log(area_log_path)
    return builtin_area_model()


@dataclass(frozen=True)
class TaskArea:
    name: str
    scale: str  # "NUM_PES" | "NUM_A_CH" | "NUM_B_CH" | "NUM_C_CH" | "TBD" | ...
    bram: float
    dsp: float
    ff: float
    lut: float
    uram: float


def _is_row_sharing_only_task(task_name: str) -> bool:
    return (
        task_name.startswith("ADD_")
        or task_name == "SSW"
        or task_name.startswith("SWB")
    )


@lru_cache(maxsize=None)
def _drdn_block_counts(num_pes: int) -> dict[str, int]:
    """
    Use crossbar.py to count how many instances of each RS-only task exist for a given NUM_PES.
    """
    try:
        from automation_tool.crossbar import CrossBarGen  # package-local
    except Exception:
        return {}

    cb = CrossBarGen(int(num_pes))
    cb.buildGraph(view=None)

    counts: dict[str, int] = {}
    for node_id in cb.graph_dict.keys():
        parts = str(node_id).split(".")
        if len(parts) >= 2:
            block = parts[1]
            counts[block] = counts.get(block, 0) + 1
    return counts


def drdn_counts(num_pes: int) -> dict[str, int]:
    return dict(_drdn_block_counts(int(num_pes)))


def max_utilization(util: dict[ResourceName, float]) -> tuple[ResourceName, float]:
    best_r: ResourceName = "LUT"
    best_v = -1.0
    for r in ("BRAM", "DSP", "FF", "LUT", "URAM"):
        v = float(util.get(r, 0.0))
        if v > best_v:
            best_v = v
            best_r = r  # type: ignore[assignment]
    return best_r, best_v


def utilization(
    used: dict[ResourceName, float],
    cap: dict[ResourceName, float],
    *,
    limit: float,
) -> dict[ResourceName, float]:
    out: dict[ResourceName, float] = {}
    for r in ("BRAM", "DSP", "FF", "LUT", "URAM"):
        denom = float(cap.get(r, 0.0)) * float(limit)
        out[r] = (float(used.get(r, 0.0)) / denom) if denom > 0 else float("inf")
    return out


def _scale_multiplier(scale: str, *, a_ch: int, b_ch: int, c_ch: int, num_pes: int) -> int:
    m = re.fullmatch(r"\s*(\d+)\s+if\s+NUM_PES\s*==\s*(\d+)\s*", scale)
    if m:
        mul = int(m.group(1))
        target = int(m.group(2))
        return mul if int(num_pes) == target else 0

    if scale == "NUM_PES":
        return int(num_pes)
    if scale == "NUM_A_CH":
        return int(a_ch)
    if scale == "NUM_B_CH":
        return int(b_ch)
    if scale == "NUM_C_CH":
        return int(c_ch)
    return 1


def estimate_kernel_area(
    tasks: list[TaskArea],
    *,
    a_ch: int,
    b_ch: int,
    c_ch: int,
    num_pes: int,
    row_sharing_capable: bool,
    group_size: int = 4,
) -> dict[ResourceName, float]:
    used: dict[ResourceName, float] = {"BRAM": 0.0, "DSP": 0.0, "FF": 0.0, "LUT": 0.0, "URAM": 0.0}

    for t in tasks:
        if (not row_sharing_capable) and _is_row_sharing_only_task(t.name):
            continue

        if row_sharing_capable and _is_row_sharing_only_task(t.name):
            cb_counts = _drdn_block_counts(int(num_pes))
            mul = int(cb_counts.get(t.name, 0))
            used["BRAM"] += mul * t.bram
            used["DSP"] += mul * t.dsp
            used["FF"] += mul * t.ff
            used["LUT"] += mul * t.lut
            used["URAM"] += mul * t.uram
            continue

        if t.name == "PEG" and t.scale == "NUM_PES":
            mul = int(num_pes // max(1, group_size))
        elif t.name.startswith("Arbiter_C_8_4"):
            mul = 1 if int(num_pes) == 80 else 0
        else:
            mul = _scale_multiplier(t.scale, a_ch=a_ch, b_ch=b_ch, c_ch=c_ch, num_pes=num_pes)

        used["BRAM"] += mul * t.bram
        used["DSP"] += mul * t.dsp
        used["FF"] += mul * t.ff
        used["LUT"] += mul * t.lut
        used["URAM"] += mul * t.uram

    return used


def parse_area_log(area_log_path: str | Path) -> tuple[dict[ResourceName, float], list[TaskArea]]:
    p = Path(area_log_path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    scale = "TBD"
    tasks: list[TaskArea] = []
    cap: dict[ResourceName, float] = {"BRAM": 0.0, "DSP": 0.0, "FF": 0.0, "LUT": 0.0, "URAM": 0.0}

    def parse_task_line(line: str) -> TaskArea | None:
        if ": BRAM:" not in line:
            return None
        name, rest = line.split(":", 1)
        parts = rest.replace("BRAM:", " BRAM:").replace("DSP:", " DSP:").replace("FF:", " FF:").replace("LUT:", " LUT:").replace("URAM:", " URAM:").split()
        vals: dict[str, float] = {}
        for i in range(0, len(parts) - 1, 2):
            k = parts[i].rstrip(":")
            try:
                vals[k] = float(parts[i + 1])
            except ValueError:
                vals[k] = 0.0
        return TaskArea(
            name=name.strip(),
            scale=scale.strip(),
            bram=float(vals.get("BRAM", 0.0)),
            dsp=float(vals.get("DSP", 0.0)),
            ff=float(vals.get("FF", 0.0)),
            lut=float(vals.get("LUT", 0.0)),
            uram=float(vals.get("URAM", 0.0)),
        )

    in_total = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("=======") and "Numbers:" in line:
            try:
                scale = line.split("Numbers:", 1)[1].strip().strip("=").strip()
            except Exception:
                scale = "TBD"
            in_total = False
            continue

        if line.startswith("The total area"):
            in_total = True
            continue

        if in_total:
            for r in ("BRAM", "DSP", "FF", "LUT", "URAM"):
                if line.startswith(r + ":"):
                    try:
                        cap[r] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        cap[r] = 0.0
            continue

        t = parse_task_line(line)
        if t is not None:
            tasks.append(t)

    return cap, tasks
