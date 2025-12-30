from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .advisor import list_runs, recommend_run, unified_pick_name
from .generate import KernelPlan, generate_kernel_sources


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    prog="python -m automation_tool",
    description="HiSpMM kernel generator (templates + DSE).",
  )
  p.add_argument("--matrix", type=Path, default=None, help="MatrixMarket .mtx path (required for DSE / --list-picks)")
  p.add_argument("--n", type=int, default=None, help="Dense N (columns of B/C) (required for DSE / --list-picks)")
  p.add_argument("--out", type=Path, default=None, help="Output directory (required for generation)")
  p.add_argument("--variant", default="all", help="Pass-through to automation_tool.dse.cycle_analysis --variant")
  p.add_argument("--resource-limit", type=float, default=1.0, help="Resource limit (1.0 = 100 percent)")
  p.add_argument("--no-resources", action="store_true", help="Disable resource analysis in advisor")
  p.add_argument("--pick", default=None, help="Force a specific label (skip RECOMMEND parsing)")
  p.add_argument("--list-picks", action="store_true", help="List available labels (supported --pick values) and exit")
  return p.parse_args()


_RE_PICK = re.compile(r"^(?P<kind>balanced|imbalanced)_(?P<stem>a(?P<a>\d+)_c(?P<c>\d+))$")


def _canonicalize_pick(pick: str) -> str:
  s = pick.strip()
  low = s.lower()
  if low == "hispmm-balanced":
    return "balanced_a10_c4"
  if low == "hispmm-imbalanced":
    return "imbalanced_a8_c4"
  return s


def _plan_from_pick(pick: str) -> KernelPlan:
  """Build a KernelPlan without running cycle_analysis (user forced pick)."""
  p = _canonicalize_pick(pick)

  # Accept unified picks like balanced_a10_c4 / imbalanced_a8_c4.
  m = _RE_PICK.match(p)
  if m:
    a_ch = int(m.group("a"))
    c_ch = int(m.group("c"))
    rs_capable = m.group("kind") == "imbalanced"
    num_pes = a_ch * 8
    # For A4/A6 points we still want stable unified pick names for misc assets.
    return KernelPlan(
      label=p,
      pick=p,
      a_ch=a_ch,
      b_ch=4,
      c_ch=c_ch,
      num_pes=num_pes,
      rs_capable=rs_capable,
    )

  # Also accept cycle_analysis legacy labels directly (mostly for debugging).
  # - a4_c8_nrs / a6_c8_nrs => balanced_a{a}_c{c}
  # - a4_c8 / a6_c8 => imbalanced_a{a}_c{c}
  # - balanced_a8_c8 / balanced_a10_c4 / imbalanced_a8_c4 already handled above
  m2 = re.match(r"^a(?P<a>\d+)_c(?P<c>\d+)(?P<nrs>_nrs)?$", p)
  if m2:
    a_ch = int(m2.group("a"))
    c_ch = int(m2.group("c"))
    rs_capable = m2.group("nrs") is None
    kind = "imbalanced" if rs_capable else "balanced"
    unified = f"{kind}_a{a_ch}_c{c_ch}"
    return KernelPlan(
      label=p,
      pick=unified,
      a_ch=a_ch,
      b_ch=4,
      c_ch=c_ch,
      num_pes=a_ch * 8,
      rs_capable=rs_capable,
    )

  raise SystemExit(
    "Unknown --pick value.\n"
    f"Got: {pick}\n"
    "Examples:\n"
    "  --pick HiSpMM-imbalanced\n"
    "  --pick imbalanced_a8_c4\n"
    "  --pick balanced_a4_c8\n"
    "  --pick HiSpMM-balanced\n"
  )


def main() -> None:
  args = _parse_args()
  repo_root = Path(__file__).resolve().parents[1]
  templates_root = repo_root / "automation_tool" / "assets" / "tasks"

  if args.list_picks:
    if args.matrix is None or args.n is None:
      raise SystemExit("--list-picks requires --matrix and --n")
    # Friendly mapping to legacy design folder names (for user orientation only).
    pick_alias = {
      "balanced_a10_c4": "HiSpMM-balanced",
      "imbalanced_a8_c4": "HiSpMM-imbalanced",
    }
    runs = list_runs(
      repo_root=repo_root,
      matrix=args.matrix,
      n=args.n,
      variant=args.variant,
      resources=not args.no_resources,
      resource_limit=args.resource_limit,
    )
    for r in runs:
      pick = unified_pick_name(r)
      if pick in pick_alias:
        print(f"{pick} ---> {pick_alias[pick]}")
      else:
        print(pick)
    return

  if args.out is None:
    raise SystemExit("Generation requires --out <output_dir>")
  out_dir = args.out

  # If user supplied --pick, do not run DSE; build plan directly.
  if args.pick is not None:
    plan = _plan_from_pick(args.pick)
  else:
    if args.matrix is None or args.n is None:
      raise SystemExit("DSE-based generation requires --matrix and --n (or provide --pick to skip DSE).")
    try:
      rec = recommend_run(
        repo_root=repo_root,
        matrix=args.matrix,
        n=args.n,
        variant=args.variant,
        resources=not args.no_resources,
        resource_limit=args.resource_limit,
        pick_label=None,
      )
    except RuntimeError as e:
      raise
    plan = KernelPlan.from_recommendation(rec, pick=unified_pick_name(rec))

  generate_kernel_sources(templates_root=templates_root, out_dir=out_dir, plan=plan)

  print(f"Wrote: {out_dir / 'src' / 'hispmm.h'}")
  print(f"Wrote: {out_dir / 'src' / 'hispmm.cpp'}")
  print(f"Wrote: {out_dir / 'src' / 'hispmm_host.cpp'}")
  kernel_type = "imbalanced" if plan.rs_capable else "balanced"
  print(f"Config: {plan.label} (kernel={kernel_type}  A_CH={plan.a_ch}  C_CH={plan.c_ch}  NUM_PES={plan.num_pes})")


if __name__ == "__main__":
  main()


