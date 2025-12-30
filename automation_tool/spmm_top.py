from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .drdn import DrdnGraph


@dataclass(frozen=True)
class TopPatch:
  rs_capable: bool
  # The DRDN graph is only used if rs_capable=True.
  drdn: DrdnGraph | None = None
  # Materialized arbiter invoke section (no preprocessor ladder).
  arbiter_invokes: str = ""


_RE_FIFO_C_SHF_BLOCK = re.compile(
  r"(?ms)^#ifdef RS_DESIGN.*?^#endif\s*$"
)

# Match the *arbiter invoke* ladder.
# It requires the block to contain an `.invoke` line.
_RE_ARBITER_INVOKE_BLOCK = re.compile(
  r"(?ms)^#if\s+NUM_PES\s*==\s*80\s*\n\s*\.invoke.*?^#endif\s*$"
)


def _render_drdn_stream_decls(drdn: DrdnGraph) -> str:
  # Declare all intermediate streams `s_*` with depth from crossbar policy.
  lines: List[str] = []
  for name in sorted(drdn.stream_depth.keys()):
    depth = drdn.stream_depth[name]
    # NOTE: intermediate streams are Cnoc_pkt by construction (SWB converts at the end).
    lines.append(f'tapa::stream<Cnoc_pkt, {depth}> {name}("{name}");')
  return "\n".join(lines)


def _render_drdn_invokes(drdn: DrdnGraph) -> str:
  # Emit one .invoke() per DRDN node in stable order.
  lines: List[str] = []
  for node in drdn.nodes:
    args = ", ".join(node.incoming + node.outgoing)
    lines.append(f"  .invoke({node.fn}, {args})")
  return "\n".join(lines)


def patch_hispmm_cpp(template: str, patch: TopPatch) -> str:
  """Patch `assets/tasks/hispmm.cpp` into a top function suitable for our config.

  Key behavior:
  - Balanced kernel (NRS-only): PEG writes Cvec_pkt directly to FIFO_C_SHF; Accumulator reads FIFO_C_SHF.
  - Imbalanced kernel (RS-capable): PEG writes Cnoc_pkt to FIFO_C_SHF; DRDN writes Cvec_pkt to FIFO_C_BUF; Accumulator reads FIFO_C_BUF.
  """

  if patch.rs_capable and patch.drdn is None:
    raise ValueError("rs_capable=True requires drdn graph.")

  out = template.rstrip() + "\n"

 
  #    Always keep FIFO_C_SHF name because crossbar.py uses it as the DRDN source.
  if patch.rs_capable:
    fifo_block = (
      'tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");\n'
      'tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_BUF("c_buf");'
    )
  else:
    fifo_block = 'tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");'

  out = _RE_FIFO_C_SHF_BLOCK.sub(fifo_block, out)

  # 2) For RS-capable: add DRDN intermediate stream declarations before tapa::task().
  if patch.rs_capable:
    assert patch.drdn is not None
    insert = _render_drdn_stream_decls(patch.drdn)
    out = out.replace("\ntapa::task()", "\n" + insert + "\n\n" + "tapa::task()", 1)

  # 3) Fix the PEG invoke output stream + Accumulator input stream.
  if patch.rs_capable:
    # PEG should write to FIFO_C_SHF (Cnoc), Accumulator should read FIFO_C_BUF (Cvec).
    out = out.replace("PEG, FIFO_A_IN, FIFO_B_IN, FIFO_B_IN, FIFO_C_SHF,",
                      "PEG, FIFO_A_IN, FIFO_B_IN, FIFO_B_IN, FIFO_C_SHF,", 1)
    out = out.replace("Accumulator, FIFO_C_ARB, FIFO_C_SHF,",
                      "Accumulator, FIFO_C_ARB, FIFO_C_BUF,", 1)
  else:
    # NRS-only: FIFO_C_SHF is already Cvec, so leave invokes intact.
    pass

  # 4) For RS-capable: inject DRDN invokes after DummyRead and before Accumulator.
  if patch.rs_capable:
    assert patch.drdn is not None
    drdn_invokes = _render_drdn_invokes(patch.drdn)
    anchor = "  .invoke<tapa::detach, NUM_B_CH>(DummyRead, FIFO_B_IN)\n"
    if anchor not in out:
      raise RuntimeError("Could not find DummyRead invoke anchor in hispmm.cpp template.")
    out = out.replace(anchor, anchor + drdn_invokes + "\n", 1)

  # 5) Replace the arbiter preprocessor ladder with a chosen invoke section.
  if patch.arbiter_invokes:
    out, n = _RE_ARBITER_INVOKE_BLOCK.subn(patch.arbiter_invokes.rstrip(), out, count=1)
    if n != 1:
      raise RuntimeError("Could not find exactly one arbiter #if/#elif/#endif block in hispmm.cpp template.")

  return out


