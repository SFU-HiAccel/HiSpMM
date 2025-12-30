from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class TaskTemplates:
  root: Path

  def read(self, name: str) -> str:
    p = self.root / name
    return p.read_text(encoding="utf-8", errors="replace")

  def exists(self, name: str) -> bool:
    return (self.root / name).exists()


def join_cpp_units(units: Iterable[str]) -> str:
  out: List[str] = []
  for u in units:
    u = u.rstrip()
    if not u:
      continue
    out.append(u)
    out.append("")  # blank line between units
  return "\n".join(out).rstrip() + "\n"


def header_banner(*, label: str) -> str:
  return f"// HiSpMM kernel configuration: {label}\n\n"


