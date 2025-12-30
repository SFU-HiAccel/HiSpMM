import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixShape:
    m: int  # rows
    k: int  # cols
    nnz: int  # effective nnz (after symmetric expansion if applicable)


def _parse_mm_banner(line: str) -> dict:
    # Example: %%MatrixMarket matrix coordinate real general
    parts = line.strip().split()
    if len(parts) < 5 or not parts[0].lower().startswith("%%matrixmarket"):
        raise ValueError("Invalid MatrixMarket banner")
    return {
        "format": parts[2].lower(),  # coordinate/array
        "field": parts[3].lower(),  # real/integer/pattern/complex
        "symmetry": parts[4].lower(),  # general/symmetric/...
    }


def load_mtx_coo_shape(path: Path) -> tuple[MatrixShape, list[tuple[int, int]]]:
    """
    Minimal MatrixMarket coordinate parser.
    Returns (shape, list_of_entries) where entries are 0-indexed (row, col).

    For 'symmetric' matrices, we expand off-diagonal entries to both (r,c) and (c,r)
    to match the host/helper behavior (they duplicate symmetric off-diagonals).
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        banner = None
        size_line = None

        def _is_comment(raw: str) -> bool:
            s = raw.lstrip()
            # MatrixMarket uses '%', but other sources commonly use '#'.
            return s.startswith("%") or s.startswith("#")

        # Read banner (if present) and find the first non-comment/non-empty line after it
        # (or, if no banner exists, treat the first non-comment line as the size line).
        for line in f:
            if not line.strip():
                continue
            if _is_comment(line):
                if banner is None and line.lower().startswith("%%matrixmarket"):
                    banner = _parse_mm_banner(line)
                continue
            size_line = line
            break

        if size_line is None:
            raise ValueError("Unexpected EOF while reading matrix header/size")

        # If no MatrixMarket banner was found, fall back to a common variant:
        # a bare coordinate header where the first data line is "M K NNZ".
        if banner is None:
            banner = {"format": "coordinate", "field": "real", "symmetry": "general"}

        try:
            m, k, _nnz_decl = map(int, size_line.strip().split()[:3])
        except Exception as e:
            raise ValueError(
                "Unrecognized matrix header/size line. Expected either:\n"
                "  - MatrixMarket banner + size line, or\n"
                "  - A bare coordinate size line: 'M K NNZ'\n"
                f"Got: {size_line.strip()!r}"
            ) from e

        if banner["format"] != "coordinate":
            raise ValueError(f"Only coordinate MatrixMarket is supported (got {banner['format']})")

        symmetric = banner["symmetry"] == "symmetric"

        entries: list[tuple[int, int]] = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            if r < 0 or c < 0:
                continue

            entries.append((r, c))
            if symmetric and r != c:
                entries.append((c, r))

        nnz_eff = len(entries)
        return MatrixShape(m=m, k=k, nnz=nnz_eff), entries