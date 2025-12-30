from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RecommendedRun:
  label: str
  a_ch: int
  c_ch: int
  num_pes: int
  mode: str  # e.g. "nrs(k)", "nrs", "rs"
  freq_mhz: float

  @property
  def row_sharing_capable(self) -> bool:
    # Convention from cycle_analysis output: "(k)" means forced by kernel (i.e., not capable).
    return not self.mode.endswith("(k)")


_RE_SUMMARY_ROW = re.compile(
  r"^\s*(?P<label>\S+)\s+"
  r"(?P<a_ch>\d+)\s+"
  r"(?P<c_ch>\d+)\s+"
  r"(?P<num_pes>\d+)\s+"
  r"(?P<mode>\S+)\s+"
  r"(?P<run_len>\S+)\s+"
  r"(?P<t1>\S+)\s+"
  r"(?P<t2_used>\S+)\s+"
  r"(?P<t3>\S+)\s+"
  r"(?P<total>\S+)\s+"
  r"(?P<freq_mhz>\S+)\s+"
  r"(?P<time_us>\S+)\s+"
  r"(?P<res_max>\S+)\s+"
  r"(?P<fits>\S+)\s*$"
)

_RE_RECOMMEND = re.compile(r"^RECOMMEND:\s+(?P<label>\S+)\s+", re.MULTILINE)


def _run_cycle_analysis(
  *,
  repo_root: Path,
  matrix: Path,
  n: int,
  variant: str,
  resources: bool,
  resource_limit: float,
) -> str:
  # Run the DSE module shipped inside this package.
  cmd = [
    sys.executable,
    "-m",
    "automation_tool.dse.cycle_analysis",
    "--matrix",
    str(matrix),
    "--n",
    str(n),
    "--variant",
    str(variant),
  ]
  if resources:
    cmd += ["--resources", "--resource-limit", str(resource_limit)]

  proc = subprocess.run(
    cmd,
    cwd=str(repo_root),
    capture_output=True,
    text=True,
  )
  if proc.returncode != 0:
    raise RuntimeError(
      "cycle_analysis failed\n"
      f"cmd: {' '.join(cmd)}\n"
      f"stdout:\n{proc.stdout}\n"
      f"stderr:\n{proc.stderr}\n"
    )
  return proc.stdout


def recommend_run(
  *,
  repo_root: Path,
  matrix: Path,
  n: int,
  variant: str = "all",
  resources: bool = True,
  resource_limit: float = 1.0,
  pick_label: Optional[str] = None,
) -> RecommendedRun:
  """Return the recommended run from cycle_analysis, or a forced `pick_label`.

  Note: `pick_label` may be a unified alias such as `balanced_a6_c8` / `imbalanced_a6_c8`.
  """
  out = _run_cycle_analysis(
    repo_root=repo_root,
    matrix=matrix,
    n=n,
    variant=variant,
    resources=resources,
    resource_limit=resource_limit,
  )

  # Parse all summary rows once and resolve aliases.
  rows = [line for line in out.splitlines() if _RE_SUMMARY_ROW.match(line)]

  def _labels_in_rows() -> set[str]:
    labels: set[str] = set()
    for line in rows:
      mm = _RE_SUMMARY_ROW.match(line)
      assert mm is not None
      labels.add(mm.group("label"))
    return labels

  def _resolve_pick_label(pick: str) -> str:
    labels = _labels_in_rows()
    if pick in labels:
      return pick

    # Legacy "design folder" aliases (user-facing).
    # These are not cycle_analysis labels; map them to canonical unified picks first.
    legacy_design_alias = {
      "hispmm-balanced": "balanced_a10_c4",
      "hispmm-imbalanced": "imbalanced_a8_c4",
    }
    norm = pick.strip()
    norm_key = norm.lower()
    if norm_key in legacy_design_alias:
      pick = legacy_design_alias[norm_key]
      if pick in labels:
        return pick

    # Unified terminology aliases:
    # - balanced_X => NRS-only variant; for legacy labels we map to "<X>_nrs" if present.
    # - imbalanced_X => RS-capable variant; for legacy labels we map to "<X>" if present.
    if pick.startswith("balanced_"):
      stem = pick[len("balanced_") :]
      candidate = stem + "_nrs"
      if candidate in labels:
        return candidate
    if pick.startswith("imbalanced_"):
      stem = pick[len("imbalanced_") :]
      if stem in labels:
        return stem

    return pick

  if pick_label is None:
    m = _RE_RECOMMEND.search(out)
    if not m:
      raise RuntimeError("Could not find 'RECOMMEND:' line in cycle_analysis output.")
    label = m.group("label")
  else:
    label = _resolve_pick_label(pick_label)

  # Parse summary row for the chosen label.
  chosen: Optional[RecommendedRun] = None
  for line in out.splitlines():
    if not line.lstrip().startswith(label + " "):
      continue
    mm = _RE_SUMMARY_ROW.match(line)
    if not mm:
      raise RuntimeError(f"Found label row but could not parse it:\n{line}")
    chosen = RecommendedRun(
      label=mm.group("label"),
      a_ch=int(mm.group("a_ch")),
      c_ch=int(mm.group("c_ch")),
      num_pes=int(mm.group("num_pes")),
      mode=mm.group("mode"),
      freq_mhz=float(mm.group("freq_mhz")),
    )
    break

  if chosen is None:
    raise RuntimeError(f"Could not find summary row for label '{label}'.")
  return chosen


def list_runs(
  *,
  repo_root: Path,
  matrix: Path,
  n: int,
  variant: str = "all",
  resources: bool = True,
  resource_limit: float = 1.0,
) -> list[RecommendedRun]:
  """List all executed runs (labels) from cycle_analysis summary table."""
  out = _run_cycle_analysis(
    repo_root=repo_root,
    matrix=matrix,
    n=n,
    variant=variant,
    resources=resources,
    resource_limit=resource_limit,
  )

  runs: list[RecommendedRun] = []
  for line in out.splitlines():
    mm = _RE_SUMMARY_ROW.match(line)
    if not mm:
      continue
    runs.append(
      RecommendedRun(
        label=mm.group("label"),
        a_ch=int(mm.group("a_ch")),
        c_ch=int(mm.group("c_ch")),
        num_pes=int(mm.group("num_pes")),
        mode=mm.group("mode"),
        freq_mhz=float(mm.group("freq_mhz")),
      )
    )
  return runs


def unified_pick_name(run: RecommendedRun) -> str:
  """Return an `automation_tool` pick name using unified terminology.

  Mapping:
  - balanced = NRS-only kernel (forced by kernel => mode endswith '(k)')
  - imbalanced = RS-capable kernel
  """
  label = run.label
  if label.startswith("balanced_") or label.startswith("imbalanced_"):
    return label

  kernel_prefix = "imbalanced" if run.row_sharing_capable else "balanced"

  # Legacy special-case: `a6_c8_nrs` should show as `balanced_a6_c8`.
  if not run.row_sharing_capable and label.endswith("_nrs"):
    label = label[: -len("_nrs")]

  return f"{kernel_prefix}_{label}"


