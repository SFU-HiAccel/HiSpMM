from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# crossbar.py is part of this package (automation_tool/crossbar.py).
from . import crossbar


@dataclass(frozen=True)
class DrdnNode:
  key: str               # e.g. "0.ADD_1.[0]"
  fn: str                # e.g. "ADD_1"
  incoming: List[str]    # e.g. ["FIFO_C_SHF[0]", "FIFO_C_SHF[1]"]
  outgoing: List[str]    # e.g. ["s_0_0", "s_1_0"]


@dataclass(frozen=True)
class DrdnGraph:
  num_pes: int
  nodes: List[DrdnNode]            # stable order
  stream_depth: Dict[str, int]     # for s_* streams


def _parse_node_fn(key: str) -> str:
  # Format from crossbar.py: "<idx>.<BLOCK>.[<level>]"
  parts = key.split(".")
  if len(parts) < 2:
    raise ValueError(f"Unexpected DRDN node key: {key}")
  return parts[1]


def _sort_key(node_key: str) -> Tuple[int, str]:
  # Sort primarily by numeric prefix, then full key for stability.
  prefix = node_key.split(".", 1)[0]
  try:
    return (int(prefix), node_key)
  except ValueError:
    return (10**9, node_key)


def build_drdn_graph(num_pes: int) -> DrdnGraph:
  g = crossbar.CrossBarGen(num_pes)
  g.buildGraph(False)
  g.computeDepth()

  nodes: List[DrdnNode] = []
  for key in sorted(g.graph_dict.keys(), key=_sort_key):
    info = g.graph_dict[key]
    nodes.append(
      DrdnNode(
        key=key,
        fn=_parse_node_fn(key),
        incoming=list(info["incoming"]),
        outgoing=list(info["outgoing"]),
      )
    )

  # crossbar.depth_dict provides per-stream dicts like {"start":..., "end":..., "depth":...}.
  stream_depth: Dict[str, int] = {}
  for k, v in g.depth_dict.items():
    name = str(k)
    if isinstance(v, dict) and "depth" in v:
      stream_depth[name] = int(v["depth"])
    else:
      # Fallback: best-effort coercion.
      stream_depth[name] = int(v)  # type: ignore[arg-type]
  return DrdnGraph(num_pes=num_pes, nodes=nodes, stream_depth=stream_depth)


def required_task_files_for_drdn(nodes: Iterable[DrdnNode]) -> List[str]:
  need = {f"{n.fn}.cpp" for n in nodes}
  # DRDN blocks always require these by construction, but keep it data-driven.
  return sorted(need)


