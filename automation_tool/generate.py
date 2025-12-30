from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

from .advisor import RecommendedRun
from .drdn import build_drdn_graph, required_task_files_for_drdn
from .misc_assets import (
  build_standalone_makefile,
  default_link_config_text,
  load_link_config_text,
  misc_config_for_pick,
  try_load_floorplan_obj,
)
from .spmm_header import SpmmDefines, patch_hispmm_h
from .spmm_top import TopPatch, patch_hispmm_cpp
from .templates import TaskTemplates, header_banner, join_cpp_units


@dataclass(frozen=True)
class KernelPlan:
  # `label` is the underlying cycle_analysis label (may be legacy like `a6_c8_nrs`).
  label: str
  # `pick` is the unified label used for misc assets (balanced/imbalanced prefixes).
  pick: str
  a_ch: int
  b_ch: int
  c_ch: int
  num_pes: int
  rs_capable: bool

  @classmethod
  def from_recommendation(cls, rec: RecommendedRun, *, pick: str) -> "KernelPlan":
    return cls(
      label=rec.label,
      pick=pick,
      a_ch=rec.a_ch,
      b_ch=4,
      c_ch=rec.c_ch,
      num_pes=rec.num_pes,
      rs_capable=rec.row_sharing_capable,
    )


def _base_task_files(*, rs_capable: bool) -> List[str]:
  base = [
    # Must appear before MM2S_* tasks.
    "inline_tasks.cpp",
    "MM2S_A.cpp",
    "MM2S_B.cpp",
    "MM2S_C.cpp",
    "DummyRead.cpp",
    "Accumulator.cpp",
    "Compute_C.cpp",
    "S2MM_C.cpp",
  ]
  base.append("PEG_RS.cpp" if rs_capable else "PEG_NRS.cpp")
  return base


def _arbiter_task_files(*, a_ch: int, c_ch: int, num_pes: int) -> List[str]:
  # This matches the selection logic already present in assets/tasks/hispmm.cpp.
  if num_pes == 80:
    return ["Arbiter_C_10_1.cpp", "Arbiter_C_8_4.cpp"]
  if num_pes == 64 and c_ch == 8:
    return ["Arbiter_C_8_1.cpp"]
  if num_pes == 64 and c_ch == 4:
    return ["Arbiter_C_16_1.cpp"]
  # Monolithic arbiter for smaller configs.
  if a_ch == 4:
    return ["Arbiter_C-a4.cpp"]
  if a_ch == 6:
    return ["Arbiter_C-a6.cpp"]
  raise ValueError(f"Unsupported arbiter configuration: A_CH={a_ch} C_CH={c_ch} NUM_PES={num_pes}")


def _arbiter_invoke_block(*, a_ch: int, c_ch: int, num_pes: int) -> str:
  # Must match the stream names declared in assets/tasks/hispmm.cpp.
  if num_pes == 80:
    return (
      "  .invoke<tapa::detach, 8>(Arbiter_C_10_1,  FIFO_C_ARB, FIFO_C_AB_INTER)\n"
      "  .invoke<tapa::detach>(Arbiter_C_8_4,  FIFO_C_AB_INTER, FIFO_C_AB)\n"
    ).rstrip()
  if num_pes == 64 and c_ch == 8:
    return "  .invoke<tapa::detach, 8>(Arbiter_C_8_1,  FIFO_C_ARB, FIFO_C_AB)".rstrip()
  if num_pes == 64 and c_ch == 4:
    return "  .invoke<tapa::detach, 4>(Arbiter_C_16_1,  FIFO_C_ARB, FIFO_C_AB)".rstrip()

  # Monolithic arbiter name is always `Arbiter_C`.
  return "  .invoke<tapa::detach>(Arbiter_C,  FIFO_C_ARB, FIFO_C_AB)".rstrip()


def generate_kernel_sources(*, templates_root: Path, out_dir: Path, plan: KernelPlan) -> None:
  """Materialize kernel sources, host sources, plus misc build assets into `out_dir/`."""
  tpl = TaskTemplates(templates_root)
  (out_dir / "src").mkdir(parents=True, exist_ok=True)

  # templates_root: <repo_root>/automation_tool/assets/tasks
  # repo_root:      <repo_root>
  repo_root = templates_root.parents[2]

  # 1) hispmm.h (kernel header)
  h_template = tpl.read("hispmm.h")
  h_out = patch_hispmm_h(
    h_template,
    SpmmDefines(
      num_a_ch=plan.a_ch,
      num_b_ch=plan.b_ch,
      num_c_ch=plan.c_ch,
      num_pes=plan.num_pes,
    ),
  )
  (out_dir / "src" / "hispmm.h").write_text(h_out, encoding="utf-8")

  # 2) hispmm.cpp (tasks + top)
  cpp_units: List[str] = []
  cpp_units.append(header_banner(label=plan.label))
  cpp_units.append('#include "hispmm.h"\n#include <cstdio>\n')

  # DRDN graph (only if RS-capable).
  drdn_graph = build_drdn_graph(plan.num_pes) if plan.rs_capable else None
  top = tpl.read("hispmm.cpp")
  top_out = patch_hispmm_cpp(
    top,
    TopPatch(
      rs_capable=plan.rs_capable,
      drdn=drdn_graph,
      arbiter_invokes=_arbiter_invoke_block(a_ch=plan.a_ch, c_ch=plan.c_ch, num_pes=plan.num_pes),
    ),
  )

  task_files: List[str] = []
  task_files += _base_task_files(rs_capable=plan.rs_capable)
  task_files += _arbiter_task_files(a_ch=plan.a_ch, c_ch=plan.c_ch, num_pes=plan.num_pes)

  if plan.rs_capable:
    assert drdn_graph is not None
    # Some SWB* tasks are thin wrappers calling templated helpers `SWB0<n>` / `SWB1<n>`.
    # crossbar.py lists SWB0_*/SWB1_* but not the underlying templates, so include them.
    if tpl.exists("SWB0.cpp"):
      task_files.append("SWB0.cpp")
    if tpl.exists("SWB1.cpp"):
      task_files.append("SWB1.cpp")
    task_files += required_task_files_for_drdn(drdn_graph.nodes)

  # Stable unique order.
  seen = set()
  ordered: List[str] = []
  for f in task_files:
    if f in seen:
      continue
    seen.add(f)
    ordered.append(f)

  for f in ordered:
    if not tpl.exists(f):
      raise FileNotFoundError(f"Missing task template in assets/tasks: {f}")
    cpp_units.append(tpl.read(f))

  cpp_units.append(top_out)
  (out_dir / "src" / "hispmm.cpp").write_text(join_cpp_units(cpp_units), encoding="utf-8")

  # 2b) Host sources (hispmm_host.cpp + common host headers)
  _generate_host_sources(repo_root=repo_root, out_dir=out_dir, plan=plan)

  # 3) Materialize build assets next to the Makefile (`make tapa`).
  # link_config.ini: always generate (TAPA expects it and Makefile passes `--connectivity`).
  try:
    link_txt = load_link_config_text(repo_root=repo_root, pick=plan.pick)
  except (KeyError, FileNotFoundError):
    # Auto-generate from channel counts (non-empty, tapac-compatible).
    link_txt = default_link_config_text(a_ch=plan.a_ch, b_ch=plan.b_ch, c_ch=plan.c_ch)
  (out_dir / "link_config.ini").write_text(link_txt.rstrip() + "\n", encoding="utf-8")

  # floorplan.json: optional. If missing, omit file and strip pre-assignments from Makefile.
  floorplan_obj = None
  try:
    floorplan_obj = try_load_floorplan_obj(repo_root=repo_root, pick=plan.pick)
  except FileNotFoundError:
    floorplan_obj = None
  floorplan_present = floorplan_obj is not None
  if floorplan_present:
    (out_dir / "floorplan.json").write_text(
      json.dumps(floorplan_obj, indent=2, separators=(",", ": ")) + "\n",
      encoding="utf-8",
    )

  cfg = misc_config_for_pick(plan.pick)
  (out_dir / "Makefile").write_text(
    build_standalone_makefile(repo_root=repo_root, cfg=cfg, floorplan_present=floorplan_present),
    encoding="utf-8",
  )

  # 4) Matrices: copy common .mtx files so `make sw-test` works out of the box.
  _copy_matrices(repo_root=repo_root, out_dir=out_dir)


def _copy_matrices(*, repo_root: Path, out_dir: Path) -> None:
  src_dir = repo_root / "automation_tool" / "assets" / "common" / "matrices"
  if not src_dir.exists():
    return
  dst_dir = out_dir / "matrices"
  dst_dir.mkdir(parents=True, exist_ok=True)
  for p in src_dir.glob("*.mtx"):
    (dst_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")


def _replace_marked_block(*, text: str, begin: str, end: str, new_body: str) -> str:
  """Replace the text between marker lines (exclusive), keeping markers."""
  lines = text.splitlines()
  out: List[str] = []
  in_block = False
  seen_begin = False
  for line in lines:
    if line.strip() == begin:
      seen_begin = True
      in_block = True
      out.append(line)
      # Insert replacement body right after begin marker.
      if new_body:
        out.extend(new_body.rstrip("\n").splitlines())
      continue
    if in_block and line.strip() == end:
      in_block = False
      out.append(line)
      continue
    if in_block:
      # Skip original block body.
      continue
    out.append(line)
  if not seen_begin:
    raise RuntimeError(f"Missing begin marker: {begin}")
  if in_block:
    raise RuntimeError(f"Missing end marker: {end}")
  return "\n".join(out).rstrip() + "\n"


def _strip_codegen_markers(text: str) -> str:
  """Remove internal marker comments from a template-expanded C++ file."""
  out_lines: List[str] = []
  for line in text.splitlines():
    s = line.strip()
    if s.startswith("// @CODEGEN:"):
      continue
    out_lines.append(line)
  return "\n".join(out_lines).rstrip() + "\n"


def _host_a_cfg(*, kernel_supports_row_sharing: bool) -> tuple[bool, str]:
  """Return (kernel_supports_row_sharing, RowSharingPolicy)."""
  # Use the DSE-selected kernel capability (plan.rs_capable) rather than inferring
  # from (A_CH, C_CH). This matters for points like A6C8/A4C8 where both RS-capable
  # and NRS-only kernels exist.
  if not kernel_supports_row_sharing:
    return (False, "hispmm_host::RowSharingPolicy::kForceDisabled")
  return (True, "hispmm_host::RowSharingPolicy::kAuto")


def _host_c_layout(*, a_ch: int) -> tuple[str, str | None]:
  """Return (CinLayout enum, RowPairing enum or None)."""
  if a_ch in (4, 6):
    return ("hispmm_host::CinLayout::LinearChunkInterleave", None)
  if a_ch == 8:
    return ("hispmm_host::CinLayout::TiledPackedRows", "hispmm_host::RowPairing::AdjacentPair")
  if a_ch == 10:
    return ("hispmm_host::CinLayout::TiledPackedRows", "hispmm_host::RowPairing::HalfGroupPair")
  return ("hispmm_host::CinLayout::LinearChunkInterleave", None)


def _generate_host_sources(*, repo_root: Path, out_dir: Path, plan: KernelPlan) -> None:
  host_root = repo_root / "automation_tool" / "assets" / "misc" / "host"
  common_dir = host_root / "common"
  main_tpl = host_root / "main" / "hispmm_host.cpp"
  if not main_tpl.exists():
    raise FileNotFoundError(f"Missing host main template: {main_tpl}")
  if not common_dir.exists():
    raise FileNotFoundError(f"Missing host common dir: {common_dir}")

  # Copy common host assets into out_dir/src/.
  for name in (
    "prepare_amt_unified.h",
    "prepare_amt_unified.cpp",
    "prepare_fpga_cin_unified.h",
    "compare_fpga_c_unified.h",
    "host_common.h",
    "mmio.h",
  ):
    src = common_dir / name
    if not src.exists():
      raise FileNotFoundError(f"Missing host common asset: {src}")
    (out_dir / "src" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

  host_text = main_tpl.read_text(encoding="utf-8")

  # Patch A config block.
  supports_rs, rs_policy = _host_a_cfg(kernel_supports_row_sharing=plan.rs_capable)
  a_block = "\n".join(
    [
      "  // Unified A-matrix packing.",
      "  hispmm_host::PrepareAmtxConfig a_cfg;",
      f"  a_cfg.kernel_supports_row_sharing = {'true' if supports_rs else 'false'};",
      f"  a_cfg.row_sharing = {rs_policy};",
      "  a_cfg.shared_row_limit = -1;",
      "  a_cfg.delta_improvement_threshold_percent = 25.0;",
      "  a_cfg.print_summary = FLAGS_verbose;",
      "",
    ]
  )
  host_text = _replace_marked_block(
    text=host_text,
    begin="// @CODEGEN:PREPARE_A_CONFIG_BEGIN",
    end="// @CODEGEN:PREPARE_A_CONFIG_END",
    new_body=a_block,
  )

  # Patch Cin config block.
  cin_layout, pairing = _host_c_layout(a_ch=plan.a_ch)
  if pairing is None:
    c_block = "\n".join(
      [
        "  // Cin/Cout packing config (LinearChunkInterleave).",
        "  hispmm_host::CinPrepareConfig c_cfg;",
        f"  c_cfg.layout = {cin_layout};",
        "  c_cfg.M1 = M1;",
        "  c_cfg.N1 = N1;",
        "  c_cfg.num_c_ch = NUM_C_CH;",
        "  c_cfg.rows_per_block = NUM_PES;",
        "  c_cfg.n0 = N0;",
        "  c_cfg.b_chunk_size = B_CHUNK_SIZE;",
        "  c_cfg.m0 = M0;",
        "",
      ]
    )
  else:
    c_block = "\n".join(
      [
        "  // Cin/Cout packing config (TiledPackedRows).",
        "  hispmm_host::CinPrepareConfig c_cfg;",
        f"  c_cfg.layout = {cin_layout};",
        f"  c_cfg.pairing = {pairing};",
        "  c_cfg.M1 = M1;",
        "  c_cfg.N1 = N1;",
        "  c_cfg.num_c_ch = NUM_C_CH;",
        "  c_cfg.rows_per_block = NUM_PES;",
        "  c_cfg.n0 = N0;",
        "  c_cfg.b_chunk_size = B_CHUNK_SIZE;",
        "  c_cfg.total_row_blocks = (M1 + NUM_PES - 1) / NUM_PES;",
        "  c_cfg.row_blocks_per_tile = M0 / NUM_PES;",
        "  c_cfg.num_tiles_m = numTilesM;",
        "  c_cfg.num_tiles_n = numTilesN;",
        "  c_cfg.chunks_per_channel = (NUM_PES / NUM_C_CH) / 2;",
        "",
      ]
    )
  host_text = _replace_marked_block(
    text=host_text,
    begin="// @CODEGEN:PREPARE_C_CONFIG_BEGIN",
    end="// @CODEGEN:PREPARE_C_CONFIG_END",
    new_body=c_block,
  )

  # Remove internal marker comments from emitted host code.
  host_text = _strip_codegen_markers(host_text)
  (out_dir / "src" / "hispmm_host.cpp").write_text(host_text, encoding="utf-8")


