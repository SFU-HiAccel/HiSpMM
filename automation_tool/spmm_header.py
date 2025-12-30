from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SpmmDefines:
  num_a_ch: int
  num_b_ch: int
  num_c_ch: int
  num_pes: int

  @property
  def log2_num_pes(self) -> int:
    # LOG_2_NUM_PES is used for masking bank id; ceil(log2) matches typical usage.
    return int(math.ceil(math.log2(self.num_pes)))


_RE_DEFINE = re.compile(r"^(?P<indent>\s*)#define\s+(?P<name>\w+)\s+(?P<rest>.*)$")


def patch_hispmm_h(template: str, defs: SpmmDefines) -> str:
  """Patch `assets/tasks/hispmm.h` into an output `hispmm.h` based on config."""
  repl = {
    "NUM_A_CH": str(defs.num_a_ch),
    "NUM_B_CH": str(defs.num_b_ch),
    "NUM_C_CH": str(defs.num_c_ch),
    "NUM_PES": str(defs.num_pes),
    "LOG_2_NUM_PES": str(defs.log2_num_pes),
  }

  out_lines = []
  for line in template.splitlines():
    m = _RE_DEFINE.match(line)
    if not m:
      out_lines.append(line)
      continue

    name = m.group("name")
    if name not in repl:
      out_lines.append(line)
      continue

    indent = m.group("indent")
    out_lines.append(f"{indent}#define {name} {repl[name]}")

  return "\n".join(out_lines).rstrip() + "\n"


