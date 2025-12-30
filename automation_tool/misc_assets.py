from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MiscConfig:
  pick: str
  floorplan_strategy: str
  max_area_limit: float = 0.95
  clock_period: float = 4.255


def _load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def default_link_config_text(*, a_ch: int, b_ch: int, c_ch: int, hbm_count: int = 32) -> str:
  """Generate a non-empty `link_config.ini` from channel counts.

  This is used when a pick is missing from the curated `link_configs.json`.
  """
  if a_ch <= 0 or b_ch <= 0 or c_ch <= 0:
    raise ValueError(f"Invalid channel counts: A_CH={a_ch} B_CH={b_ch} C_CH={c_ch}")

  # Simple contiguous bank assignment policy.
  # Any valid mapping is acceptable for tapac; curated entries may be better for performance.
  bank = 0
  lines = ["[connectivity]", ""]

  # B_in
  for i in range(b_ch):
    if bank >= hbm_count:
      raise ValueError("HBM bank allocation exceeded device capacity while assigning B_in.")
    lines.append(f"sp=hispmm.B_in_{i}:HBM[{bank}]")
    bank += 1

  # A
  for i in range(a_ch):
    if bank >= hbm_count:
      raise ValueError("HBM bank allocation exceeded device capacity while assigning A.")
    lines.append(f"sp=hispmm.A_{i}:HBM[{bank}]")
    bank += 1

  # C_in
  for i in range(c_ch):
    if bank >= hbm_count:
      raise ValueError("HBM bank allocation exceeded device capacity while assigning C_in.")
    lines.append(f"sp=hispmm.C_in_{i}:HBM[{bank}]")
    bank += 1

  # C_out
  for i in range(c_ch):
    if bank >= hbm_count:
      raise ValueError("HBM bank allocation exceeded device capacity while assigning C_out.")
    lines.append(f"sp=hispmm.C_out_{i}:HBM[{bank}]")
    bank += 1

  return "\n".join(lines).rstrip() + "\n"


def load_link_config_text(*, repo_root: Path, pick: str) -> str:
  p = repo_root / "automation_tool" / "assets" / "misc" / "connectivity" / "link_configs.json"
  obj = _load_json(p)
  table = obj.get("link_configs", {})
  if pick not in table:
    raise KeyError(f"Missing link_config for pick '{pick}' in {p}")
  txt = table[pick]
  if not isinstance(txt, str):
    raise TypeError(f"link_configs['{pick}'] must be a string (raw .ini content)")
  return txt.rstrip() + "\n"


def try_load_floorplan_obj(*, repo_root: Path, pick: str) -> Optional[dict]:
  p = repo_root / "automation_tool" / "assets" / "misc" / "floorplans" / "floorplans.json"
  obj = _load_json(p)
  table = obj.get("floorplans", {})
  if pick not in table:
    return None
  fp = table[pick]
  if not isinstance(fp, dict):
    raise TypeError(f"floorplans['{pick}'] must be a JSON object")
  return fp


def misc_config_for_pick(pick: str) -> MiscConfig:
  # Keep it explicit.
  if pick == "balanced_a10_c4":
    return MiscConfig(pick=pick, floorplan_strategy="SLR_LEVEL_FLOORPLANNING")
  if pick in ("balanced_a8_c8", "imbalanced_a8_c4"):
    return MiscConfig(pick=pick, floorplan_strategy="HALF_SLR_LEVEL_FLOORPLANNING")
  # Fallback: HSLR is the safer default for most configs.
  return MiscConfig(pick=pick, floorplan_strategy="HALF_SLR_LEVEL_FLOORPLANNING")


def _strip_missing_floorplan_flags_from_common_mk(common_mk: str) -> str:
  """When no floorplan.json is available:

  We just omit pre-assignments when we don't have a file.

  Also omit `--max-area-limit` so TAPA can decide area limits itself.
  """
  drop_substrings = [
    "--floorplan-pre-assignments",
    "--max-area-limit",
  ]
  out_lines = []
  for line in common_mk.splitlines():
    if any(s in line for s in drop_substrings):
      continue
    out_lines.append(line)
  return "\n".join(out_lines).rstrip() + "\n"


def _strip_common_mk_for_standalone(common_mk: str) -> str:
  """Strip lines from common.mk that become redundant in a generated standalone Makefile."""
  drop_prefixes = (
    ".PHONY:",
    "CONNECTIVITY_FILE ?=",
    "FLOORPLAN_PREASSIGNMENTS ?=",
    "FLOORPLAN_STRATEGY ?=",
    "MAX_AREA_LIMIT ?=",
    "CLOCK_PERIOD ?=",
  )
  out_lines = []
  for line in common_mk.splitlines():
    s = line.strip()
    if not s:
      out_lines.append(line)
      continue
    if s.startswith(drop_prefixes):
      continue
    if s.startswith("# Defaults (so this file can be included"):
      continue
    out_lines.append(line)
  return "\n".join(out_lines).rstrip() + "\n"


def build_standalone_makefile(*, repo_root: Path, cfg: MiscConfig, floorplan_present: bool) -> str:
  """Create a single Makefile by prepending config vars to assets/misc/Makefile/common.mk."""
  common_path = repo_root / "automation_tool" / "assets" / "misc" / "Makefile" / "common.mk"
  common = common_path.read_text(encoding="utf-8").rstrip() + "\n"
  common = _strip_common_mk_for_standalone(common)
  if not floorplan_present:
    common = _strip_missing_floorplan_flags_from_common_mk(common)

  header = (
    "##\n"
    f"## Makefile for HiSpMM configuration: {cfg.pick}\n"
    "##\n"
    "## Expected files next to this Makefile:\n"
    "## - link_config.ini\n"
    + ("## - floorplan.json\n" if floorplan_present else "## - (no floorplan)\n")
    + "##\n"
    + "\n"
    + ".PHONY: host tapa hw-build hw-test sw-test clean\n"
    + "\n"
    + "CONNECTIVITY_FILE := link_config.ini\n"
    + (f"FLOORPLAN_STRATEGY := {cfg.floorplan_strategy}\n")
    + ("FLOORPLAN_PREASSIGNMENTS := floorplan.json\n" if floorplan_present else "")
    + (f"MAX_AREA_LIMIT := {cfg.max_area_limit}\n" if floorplan_present else "")
    + f"CLOCK_PERIOD := {cfg.clock_period}\n"
    + "\n"
  )
  return header + common


