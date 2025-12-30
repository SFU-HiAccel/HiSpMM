def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1e6 or (ax > 0 and ax < 1e-3):
        return f"{x:.3g}"
    return f"{x:.1f}"


def short_mode(mode: str) -> str:
    if mode.startswith("no_row_sharing"):
        if "forced_by_kernel" in mode:
            return "nrs(k)"
        if "forced" in mode:
            return "nrs(f)"
        return "nrs"
    if mode.startswith("row_sharing"):
        if "forced_by_kernel" in mode:
            return "rs(k)"
        if "forced" in mode:
            return "rs(f)"
        return "rs"
    return mode


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def print_row(cells: list[str]) -> None:
        print("  " + "  ".join(cells[i].ljust(widths[i]) for i in range(len(cells))))

    print_row(header)
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print_row(row)

def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1e6 or (ax > 0 and ax < 1e-3):
        return f"{x:.3g}"
    return f"{x:.1f}"


def short_mode(mode: str) -> str:
    if mode.startswith("no_row_sharing"):
        if "forced_by_kernel" in mode:
            return "nrs(k)"
        if "forced" in mode:
            return "nrs(f)"
        return "nrs"
    if mode.startswith("row_sharing"):
        if "forced_by_kernel" in mode:
            return "rs(k)"
        if "forced" in mode:
            return "rs(f)"
        return "rs"
    return mode


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def print_row(cells: list[str]) -> None:
        print("  " + "  ".join(cells[i].ljust(widths[i]) for i in range(len(cells))))

    print_row(header)
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print_row(row)


