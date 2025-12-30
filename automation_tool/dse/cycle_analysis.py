"""
Cycle count estimator for HiSpMM.

Executed by automation_tool via:
  python -m automation_tool.dse.cycle_analysis --matrix ... --n ... [other flags]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cycle_models import cycle_terms, total_sum
from .load_models import (
    compute_delta_with_row_sharing_auto,
    compute_term2_from_runlen_tiled,
    compute_term2_tiled,
)
from .mm_parser import load_mtx_coo_shape
from .reporting import fmt_num, print_table, short_mode
from .resource_models import (
    drdn_counts,
    estimate_kernel_area,
    load_area_model,
    max_utilization,
    utilization,
)


FREQ_MHZ_BY_LABEL: dict[str, float] = {
    "balanced_a10_c4": 216.0,
    "balanced_a8_c8": 204.0,
    "a6_c8": 225.0,
    "a6_c8_nrs": 225.0,
    "a4_c8": 225.0,
    "a4_c8_nrs": 225.0,
    "imbalanced_a8_c4": 225.0,
    # Sweep labels (derived from the measured points above)
    "nrs_a10_c4": 216.0,
    "nrs_a8_c8": 204.0,
}


def _freq_mhz_for_run(*, label: str, default: float) -> float:
    if label in FREQ_MHZ_BY_LABEL:
        return float(FREQ_MHZ_BY_LABEL[label])
    if label.startswith("nrs_a10_"):
        return 216.0
    if label == "nrs_a8_c8":
        return 204.0
    return float(default)


def _build_runs(variant: str, c_ch: int) -> list[tuple[str, int, int, bool, str | None]]:
    # Each run: (label, A_CH, C_CH, can_row_share, force_load_model)
    if variant == "balanced":
        return [(f"balanced_a10_c{c_ch}", 10, c_ch, False, None)]
    if variant == "imbalanced":
        return [(f"imbalanced_a8_c{c_ch}", 8, c_ch, True, None)]
    if variant == "both":
        return [
            (f"balanced_a10_c{c_ch}", 10, c_ch, False, None),
            (f"imbalanced_a8_c{c_ch}", 8, c_ch, True, None),
        ]
    if variant == "sweep":
        runs: list[tuple[str, int, int, bool, str | None]] = []
        for a_ch in (4, 6, 8, 10):
            for c_ch_run in (4, 8):
                # NRS sweep
                if not (a_ch == 10 and c_ch_run == 8):
                    runs.append((f"nrs_a{a_ch}_c{c_ch_run}", a_ch, c_ch_run, False, None))
                # RS sweep (exclude impractical points)
                if not (
                    (a_ch == 10 and c_ch_run == 8)
                    or (a_ch == 10 and c_ch_run == 4)
                    or (a_ch == 8 and c_ch_run == 8)
                ):
                    runs.append((f"rs_a{a_ch}_c{c_ch_run}", a_ch, c_ch_run, True, None))
        return runs
    # all
    return [
        (f"balanced_a10_c{c_ch}", 10, c_ch, False, None),
        ("balanced_a8_c8", 8, 8, False, None),
        ("a6_c8", 6, 8, True, None),
        ("a6_c8_nrs", 6, 8, False, None),
        ("a4_c8", 4, 8, True, None),
        ("a4_c8_nrs", 4, 8, False, None),
        (f"imbalanced_a8_c{c_ch}", 8, c_ch, True, None),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cycle count estimator for HiSpMM equation.")
    ap.add_argument("--matrix", required=True, help="Path to input MatrixMarket .mtx (sparse A)")
    ap.add_argument("--n", type=int, required=True, help="N (dense B columns)")
    ap.add_argument("--k0", type=int, default=4096)
    ap.add_argument("--n0", type=int, default=8)
    ap.add_argument("--b-ch", type=int, default=4)
    ap.add_argument("--c-ch", type=int, default=4)
    ap.add_argument("--variant", choices=["balanced", "imbalanced", "both", "all", "sweep"], default="both")
    ap.add_argument("--pes-per-ch", type=int, default=8)
    ap.add_argument("--ii-dist", type=int, default=8)
    ap.add_argument("--m0", type=int, help="M0 override. If omitted, uses A_CH * PES_PER_CH * 8192.")
    ap.add_argument("--load-model", choices=["no_row_sharing", "row_sharing", "auto"], default="auto")
    ap.add_argument("--verbose", action="store_true", help="Print detailed per-run logs.")
    ap.add_argument("--term2-model", choices=["runlen", "nnz_delta"], default="runlen")
    ap.add_argument(
        "--resources",
        action="store_true",
        help="Enable resource estimation tables (also enabled automatically when --resource-limit != 1.0 or --area-log is provided).",
    )
    ap.add_argument("--area-log", default=None, help="Optional override path to area.log for resource estimation.")
    ap.add_argument("--resource-limit", type=float, default=1.0)
    ap.add_argument("--freq-default", type=float, default=225.0)
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    shape, entries = load_mtx_coo_shape(matrix_path)
    rho = shape.nnz / (shape.m * shape.k) if (shape.m * shape.k) != 0 else 0.0

    runs = _build_runs(args.variant, args.c_ch)

    totals_rows: list[dict] = []

    resources_enabled = bool(args.resources) or (args.area_log is not None) or (float(args.resource_limit) != 1.0)
    area_cap = None
    area_tasks = None
    if resources_enabled:
        area_cap, area_tasks = load_area_model(args.area_log)

    for label, a_ch, c_ch_run, can_row_share, force_load_model in runs:
        m0 = args.m0 if args.m0 is not None else a_ch * args.pes_per_ch * 8192
        num_pes = a_ch * args.pes_per_ch
        depth_eff = min(m0, shape.m) if shape.m > 0 else m0

        delta1, delta2, delta_imp, chosen = compute_delta_with_row_sharing_auto(
            shape=shape,
            entries=entries,
            depth=depth_eff,
            window=args.k0,
            num_pes=num_pes,
            ii_dist=args.ii_dist,
            padding=1,
            shared_row_limit=depth_eff // 2,
        )

        if not can_row_share:
            delta = delta1
            chosen_mode = "no_row_sharing(forced_by_kernel)"
        elif force_load_model == "no_row_sharing":
            delta = delta1
            chosen_mode = "no_row_sharing(forced_by_run)"
        elif force_load_model == "row_sharing":
            delta = delta2
            chosen_mode = "row_sharing(forced_by_run)"
        elif args.load_model == "no_row_sharing":
            delta = delta1
            chosen_mode = "no_row_sharing(forced)"
        elif args.load_model == "row_sharing":
            delta = delta2
            chosen_mode = "row_sharing(forced)"
        else:
            delta = delta2 if chosen == "row_sharing" else delta1
            chosen_mode = chosen

        nrs_penalty = 1.03 if (can_row_share and chosen_mode == "no_row_sharing") else 1.0

        term2_mode = "row_sharing" if chosen_mode.startswith("row_sharing") else "no_row_sharing"
        nnz_delta = compute_term2_tiled(
            shape=shape,
            entries=entries,
            depth=depth_eff,
            window=args.k0,
            num_pes=num_pes,
            ii_dist=args.ii_dist,
            n=args.n,
            n0=args.n0,
            mode=term2_mode,
            padding=1,
            shared_row_limit=depth_eff // 2,
        )["t2"]
        runlen_out = compute_term2_from_runlen_tiled(
            shape=shape,
            entries=entries,
            depth=depth_eff,
            window=args.k0,
            num_pes=num_pes,
            ii_dist=args.ii_dist,
            n=args.n,
            n0=args.n0,
            mode=term2_mode,
            padding=1,
            shared_row_limit=depth_eff // 2,
        )
        run_len = int(runlen_out["run_len"])
        t2_runlen = float(runlen_out["t2_runlen"])
        t2_used = t2_runlen if args.term2_model == "runlen" else nnz_delta

        terms = cycle_terms(
            m=shape.m,
            k=shape.k,
            n=args.n,
            rho=rho,
            k0=args.k0,
            m0=m0,
            b_ch=args.b_ch,
            c_ch=c_ch_run,
            a_ch=a_ch,
            pes_per_ch=args.pes_per_ch,
            n0=args.n0,
            delta_load=delta,
            t2_used=t2_used,
        )
        total = total_sum(t1=terms["t1"], t2_used=terms["t2"], t3=terms["t3"]) * nrs_penalty
        freq_mhz = _freq_mhz_for_run(label=label, default=args.freq_default)
        time_us = (total / freq_mhz) if freq_mhz > 0 else float("inf")

        res_max_str = ""
        res_ok = True
        util_abs = None
        drdn = None
        if area_cap is not None and area_tasks is not None:
            used = estimate_kernel_area(
                area_tasks,
                a_ch=a_ch,
                b_ch=args.b_ch,
                c_ch=c_ch_run,
                num_pes=num_pes,
                row_sharing_capable=can_row_share,
            )
            util_abs = utilization(used, area_cap, limit=1.0)
            max_r, max_v_abs = max_utilization(util_abs)
            res_max_str = f"{max_r}:{max_v_abs*100.0:.1f}%"
            res_ok = max_v_abs <= float(args.resource_limit)
            drdn = drdn_counts(num_pes) if can_row_share else {}

        totals_rows.append(
            {
                "label": label,
                "a_ch": a_ch,
                "c_ch": c_ch_run,
                "num_pes": num_pes,
                "mode": chosen_mode,
                "t1": terms["t1"],
                "t2_used": terms["t2"],
                "t3": terms["t3"],
                "total": total,
                "freq_mhz": freq_mhz,
                "time_us": time_us,
                "run_len": run_len,
                "res_max": res_max_str,
                "res_ok": res_ok,
                "util_abs": util_abs,
                "drdn": drdn,
            }
        )

        if args.verbose:
            print(f"\n=== Run: {label} ===")
            print(f"A_CH={a_ch}  C_CH={c_ch_run}  NUM_PES={num_pes}  M0={m0}")
            print(f"delta1={delta1:.6g} delta2={delta2:.6g} improvement={delta_imp:.2f}%  used={delta:.6g} mode={chosen_mode}")
            print(f"term2_used({args.term2_model})={t2_used:.6g}  (run_len={run_len} t2_runlen={t2_runlen:.6g} t2_nnz_delta={nnz_delta:.6g})")
            print(f"t1={terms['t1']:.6g} t3={terms['t3']:.6g}")
            print(f"TOTAL(sum)={total:.6g}")
            print(f"freq_MHz={freq_mhz:.3f}")
            if area_cap is not None and area_tasks is not None:
                print(f"resource_limit={args.resource_limit:.3g}  res_max={res_max_str}  fits={res_ok}")

    # Prefer lower time, and if tied prefer larger A_CH.
    if area_cap is not None:
        totals_rows.sort(
            key=lambda r: (
                not bool(r.get("res_ok", True)),
                float(r["time_us"]),
                -int(r["a_ch"]),
            )
        )
    else:
        totals_rows.sort(key=lambda r: (float(r["time_us"]), -int(r["a_ch"])))

    header = ["label", "A_CH", "C_CH", "NUM_PES", "mode", "run_len", "t1", "t2_used", "t3", "TOTAL", "freq_MHz", "time_us"]
    if area_cap is not None:
        header += ["res_max", "fits"]
    rows: list[list[str]] = []
    for r in totals_rows:
        row = [
            r["label"],
            str(r["a_ch"]),
            str(r["c_ch"]),
            str(r["num_pes"]),
            short_mode(r["mode"]),
            str(int(r["run_len"])),
            fmt_num(float(r["t1"])),
            fmt_num(float(r["t2_used"])),
            fmt_num(float(r["t3"])),
            fmt_num(float(r["total"])),
            fmt_num(float(r["freq_mhz"])),
            fmt_num(float(r["time_us"])),
        ]
        if area_cap is not None:
            row += [str(r.get("res_max", "")), ("Y" if r.get("res_ok", True) else "N")]
        rows.append(row)

    print("\n=== Summary (all runs) ===")
    print_table(header, rows)

    eligible = totals_rows
    if area_cap is not None:
        eligible = [r for r in totals_rows if r.get("res_ok", True)]
        if not eligible:
            eligible = totals_rows
            print("\nWARNING: No configs fit the resource limit; recommending best by time_us anyway.")

    # Prefer lower time, and if tied prefer larger A_CH.
    best = min(eligible, key=lambda x: (float(x["time_us"]), -int(x["a_ch"])))
    print("\n=== Recommendation (min TOTAL among executed runs) ===")
    print(
        f"RECOMMEND: {best['label']}  "
        f"(A_CH={best['a_ch']}  C_CH={best['c_ch']}  NUM_PES={best['num_pes']}  mode={best['mode']})"
    )
    print(f"min time_us={best['time_us']:.6g}  (TOTAL={best['total']:.6g} @ {best['freq_mhz']:.3f} MHz)")


if __name__ == "__main__":
    main()


