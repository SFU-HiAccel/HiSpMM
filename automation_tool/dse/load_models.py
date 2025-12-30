import math

from .mm_parser import MatrixShape


def _tile_dims(*, m: int, k: int, depth: int, window: int, tr: int, tc: int) -> tuple[int, int]:
    """Packed (non-padded) tile dimensions for the last partial tiles."""
    tm = m - tr * depth
    tk = k - tc * window
    tile_m = tm if tm < depth else depth
    tile_k = tk if tk < window else window
    if tile_m < 0:
        tile_m = 0
    if tile_k < 0:
        tile_k = 0
    return (tile_m, tile_k)


def _build_tile_row_sizes(
    *,
    shape: MatrixShape,
    entries: list[tuple[int, int]],
    depth: int,
    window: int,
) -> dict[tuple[int, int], dict[int, int]]:
    """
    Build per-tile rowSizes stored sparsely:
      tiles[(tr, tc)] = {local_row_id: nnz_in_that_row_in_tile, ...}
    """
    tiles: dict[tuple[int, int], dict[int, int]] = {}
    for (r, c) in entries:
        if r >= shape.m or c >= shape.k:
            continue
        tr = r // depth
        tc = c // window
        lr = r % depth
        key = (tr, tc)
        d = tiles.get(key)
        if d is None:
            d = {}
            tiles[key] = d
        d[lr] = d.get(lr, 0) + 1
    return tiles


def compute_delta_no_row_sharing(
    *,
    shape: MatrixShape,
    entries: list[tuple[int, int]],
    depth: int,
    window: int,
    num_pes: int,
    ii_dist: int,
) -> float:
    """Overall delta (stddev/mean) aggregated across all tiles and PEs."""
    if shape.m <= 0 or shape.k <= 0:
        return 0.0
    num_tiles_rows = (shape.m + depth - 1) // depth
    num_tiles_cols = (shape.k + window - 1) // window
    tiles = _build_tile_row_sizes(shape=shape, entries=entries, depth=depth, window=window)

    load_sum = 0.0
    load_sq_sum = 0.0
    load_count = 0
    for tr in range(num_tiles_rows):
        for tc in range(num_tiles_cols):
            row_dict = tiles.get((tr, tc), {})
            sorted_rows = sorted(row_dict.items(), key=lambda x: x[1], reverse=True)
            loads = [[0] * ii_dist for _ in range(num_pes)]
            for row_id, row_size in sorted_rows:
                pe = row_id % num_pes
                # place into least-loaded lane
                min_idx = 0
                min_val = loads[pe][0]
                for ii in range(1, ii_dist):
                    v = loads[pe][ii]
                    if v < min_val:
                        min_val = v
                        min_idx = ii
                loads[pe][min_idx] += row_size
            for pe in range(num_pes):
                pe_load = sum(loads[pe])
                load_sum += float(pe_load)
                load_sq_sum += float(pe_load) * float(pe_load)
                load_count += 1

    if load_count == 0:
        return 0.0
    mean = load_sum / load_count
    if mean < 1.0:
        mean = 1.0
    var = load_sq_sum / load_count - mean * mean
    if var < 0.0:
        var = 0.0
    return math.sqrt(var) / mean


def _balance_workload_shared_rows_for_tile(
    *,
    row_dict: dict[int, int],
    depth: int,
    num_pes: int,
    shared_row_limit: int,
    padding: int,
) -> tuple[list[int], int]:
    """Port of the imbalanced helper's per-tile shared-row selection logic."""
    pe_work = [0] * num_pes
    total = 0
    for row_id, row_size in row_dict.items():
        pe_work[row_id % num_pes] += row_size
        total += row_size

    if total == 0:
        return ([], padding)

    max_val = max(pe_work)
    if max_val < 2:
        return ([], max_val + padding)

    sorted_rows = sorted(row_dict.items(), key=lambda x: x[1], reverse=True)
    scheduled = (max_val + padding) * num_pes
    imb = (scheduled - total) / total
    extra_cycles = 0
    removed: list[int] = []

    limit = min(shared_row_limit, len(sorted_rows))
    for ii in range(limit):
        row_id, row_size = sorted_rows[ii]
        new_total = total - row_size
        new_pe_work = pe_work[:]
        new_pe_work[row_id % num_pes] -= row_size
        new_max = max(new_pe_work)
        scheduled = num_pes * new_max
        new_imb = (scheduled - new_total) / new_total if new_total != 0 else float("inf")

        if (imb - new_imb) > 0 or ii < 2:
            total = new_total
            pe_work = new_pe_work
            max_val = new_max
            extra_cycles += ((row_size - 1) // num_pes) + 1
            removed.append(row_id)

        if abs(imb - new_imb) <= 0.01 and new_imb < 2:
            break
        imb = new_imb

    removed.sort()
    max_pe_load1 = max_val + extra_cycles + padding
    return (removed, max_pe_load1)


def compute_delta_with_row_sharing_auto(
    *,
    shape: MatrixShape,
    entries: list[tuple[int, int]],
    depth: int,
    window: int,
    num_pes: int,
    ii_dist: int,
    padding: int = 1,
    shared_row_limit: int | None = None,
) -> tuple[float, float, float, str]:
    """Compute (delta1, delta2, improvement%, chosen_mode)."""
    if shared_row_limit is None:
        shared_row_limit = depth // 2

    num_tiles_rows = (shape.m + depth - 1) // depth
    num_tiles_cols = (shape.k + window - 1) // window
    tiles = _build_tile_row_sizes(shape=shape, entries=entries, depth=depth, window=window)

    delta1 = compute_delta_no_row_sharing(
        shape=shape, entries=entries, depth=depth, window=window, num_pes=num_pes, ii_dist=ii_dist
    )

    shared_rows: dict[tuple[int, int], list[int]] = {}
    for tr in range(num_tiles_rows):
        for tc in range(num_tiles_cols):
            row_dict = tiles.get((tr, tc), {})
            removed, _ = _balance_workload_shared_rows_for_tile(
                row_dict=row_dict,
                depth=depth,
                num_pes=num_pes,
                shared_row_limit=shared_row_limit,
                padding=padding,
            )
            if removed:
                shared_rows[(tr, tc)] = removed

    # Aggregate per-PE loads over all tiles using the RS scheduling rule.
    load_sum = 0.0
    load_sq_sum = 0.0
    load_count = 0
    for tr in range(num_tiles_rows):
        for tc in range(num_tiles_cols):
            row_dict = tiles.get((tr, tc), {})
            removed = shared_rows.get((tr, tc), [])
            removed_set = set(removed)
            loads = [[0] * ii_dist for _ in range(num_pes)]

            shared_sorted = sorted(((r, row_dict.get(r, 0)) for r in removed), key=lambda x: x[1], reverse=True)
            for row_id, row_size in shared_sorted:
                if row_size <= 0:
                    continue
                load_size = ((row_size - 1) // num_pes) + 1
                for pe in range(num_pes):
                    min_idx = 0
                    min_val = loads[pe][0]
                    for ii in range(1, ii_dist):
                        v = loads[pe][ii]
                        if v < min_val:
                            min_val = v
                            min_idx = ii
                    loads[pe][min_idx] += load_size

            rem_rows = [(r, sz) for (r, sz) in row_dict.items() if r not in removed_set]
            rem_rows.sort(key=lambda x: x[1], reverse=True)
            for row_id, row_size in rem_rows:
                pe = row_id % num_pes
                min_idx = 0
                min_val = loads[pe][0]
                for ii in range(1, ii_dist):
                    v = loads[pe][ii]
                    if v < min_val:
                        min_val = v
                        min_idx = ii
                loads[pe][min_idx] += row_size

            for pe in range(num_pes):
                pe_load = sum(loads[pe])
                load_sum += float(pe_load)
                load_sq_sum += float(pe_load) * float(pe_load)
                load_count += 1

    if load_count == 0:
        delta2 = 0.0
    else:
        mean = load_sum / load_count
        if mean < 1.0:
            mean = 1.0
        var = load_sq_sum / load_count - mean * mean
        if var < 0.0:
            var = 0.0
        delta2 = math.sqrt(var) / mean

    imp = ((delta1 - delta2) / delta1 * 100.0) if delta1 > 0.0 else 0.0
    chosen = "row_sharing" if imp >= 25.0 else "no_row_sharing"
    return (delta1, delta2, imp, chosen)


def compute_term2_tiled(
    *,
    shape: MatrixShape,
    entries: list[tuple[int, int]],
    depth: int,
    window: int,
    num_pes: int,
    ii_dist: int,
    n: int,
    n0: int,
    mode: str,
    padding: int = 1,
    shared_row_limit: int | None = None,
) -> dict:
    """Tile-based term2 using nnz+delta."""
    if shared_row_limit is None:
        shared_row_limit = depth // 2
    num_tiles_rows = (shape.m + depth - 1) // depth
    num_tiles_cols = (shape.k + window - 1) // window
    tiles = _build_tile_row_sizes(shape=shape, entries=entries, depth=depth, window=window)

    shared_rows: dict[tuple[int, int], list[int]] = {}
    if mode == "row_sharing":
        for tr in range(num_tiles_rows):
            for tc in range(num_tiles_cols):
                row_dict = tiles.get((tr, tc), {})
                removed, _ = _balance_workload_shared_rows_for_tile(
                    row_dict=row_dict,
                    depth=depth,
                    num_pes=num_pes,
                    shared_row_limit=shared_row_limit,
                    padding=padding,
                )
                if removed:
                    shared_rows[(tr, tc)] = removed

    factor_bn = (n / n0) if n0 != 0 else 0.0
    t2 = 0.0
    tiles_out: list[dict] = []
    for tr in range(num_tiles_rows):
        for tc in range(num_tiles_cols):
            row_dict = tiles.get((tr, tc), {})
            if mode == "no_row_sharing":
                tile_nnz = sum(row_dict.values())
                tile_delta, _ = (0.0, tile_nnz)
                # quick delta: use stddev/mean estimator per tile
                tile_delta, tile_nnz = (0.0, tile_nnz) if tile_nnz == 0 else (compute_delta_no_row_sharing(shape=shape, entries=entries, depth=depth, window=window, num_pes=num_pes, ii_dist=ii_dist), tile_nnz)
            else:
                removed = shared_rows.get((tr, tc), [])
                tile_delta = 0.0
                tile_nnz = sum(row_dict.values())
                if tile_nnz > 0:
                    # reuse the per-tile delta implementation
                    tile_delta, tile_nnz = (0.0, tile_nnz) if not removed else (tile_delta, tile_nnz)

            tile_t2 = (tile_nnz / num_pes) * factor_bn * (1.0 + tile_delta) if num_pes != 0 else 0.0
            t2 += tile_t2
            tiles_out.append({"tr": tr, "tc": tc, "tile_nnz": tile_nnz, "delta": tile_delta, "t2": tile_t2})

    return {"t2": t2, "tiles": tiles_out}


def compute_term2_from_runlen_tiled(
    *,
    shape: MatrixShape,
    entries: list[tuple[int, int]],
    depth: int,
    window: int,
    num_pes: int,
    ii_dist: int,
    n: int,
    n0: int,
    mode: str,
    padding: int = 1,
    shared_row_limit: int | None = None,
) -> dict:
    """Term2 using helper run length (run_len * (N / N0))."""
    if shared_row_limit is None:
        shared_row_limit = depth // 2
    num_tiles_rows = (shape.m + depth - 1) // depth
    num_tiles_cols = (shape.k + window - 1) // window
    tiles = _build_tile_row_sizes(shape=shape, entries=entries, depth=depth, window=window)

    # Determine shared rows per tile if needed.
    shared_rows: dict[tuple[int, int], list[int]] = {}
    if mode == "row_sharing":
        for tr in range(num_tiles_rows):
            for tc in range(num_tiles_cols):
                row_dict = tiles.get((tr, tc), {})
                removed, _ = _balance_workload_shared_rows_for_tile(
                    row_dict=row_dict,
                    depth=depth,
                    num_pes=num_pes,
                    shared_row_limit=shared_row_limit,
                    padding=padding,
                )
                if removed:
                    shared_rows[(tr, tc)] = removed

    # Run-length proxy matching the legacy host model:
    # - Build Loads[NUM_PES][II_DIST]
    # - Schedule rows in descending size:
    #   - no_row_sharing: row goes to pe=(row_id % NUM_PES), placed in least-loaded lane of that PE
    #   - row_sharing: selected "shared" rows are split across all PEs with load_size=ceil(row_size/NUM_PES)
    #     and placed in least-loaded lane for each PE; remaining rows use the no-RS rule.
    def tile_runlen_no_rs(row_dict: dict[int, int]) -> int:
        sorted_rows = sorted(row_dict.items(), key=lambda x: x[1], reverse=True)
        loads = [[0] * ii_dist for _ in range(num_pes)]
        for row_id, row_size in sorted_rows:
            pe = row_id % num_pes
            min_idx = min(range(ii_dist), key=lambda ii: loads[pe][ii])
            loads[pe][min_idx] += row_size
        max_bucket = max((loads[pe][ii] for pe in range(num_pes) for ii in range(ii_dist)), default=0)
        return int((max_bucket + padding) * ii_dist)

    def tile_runlen_row_sharing(row_dict: dict[int, int], removed: list[int]) -> int:
        removed_set = set(removed)
        loads = [[0] * ii_dist for _ in range(num_pes)]

        # Schedule shared rows first (largest to smallest), split across all PEs.
        shared_sorted = sorted(
            ((r, row_dict.get(r, 0)) for r in removed),
            key=lambda x: x[1],
            reverse=True,
        )
        for _row_id, row_size in shared_sorted:
            if row_size <= 0:
                continue
            load_size = ((row_size - 1) // num_pes) + 1  # ceil(row_size / num_pes)
            for pe in range(num_pes):
                min_idx = min(range(ii_dist), key=lambda ii: loads[pe][ii])
                loads[pe][min_idx] += load_size

        # Schedule remaining rows (largest to smallest) to their owning PE.
        rem_rows = [(r, sz) for (r, sz) in row_dict.items() if r not in removed_set]
        rem_rows.sort(key=lambda x: x[1], reverse=True)
        for row_id, row_size in rem_rows:
            if row_size <= 0:
                continue
            pe = row_id % num_pes
            min_idx = min(range(ii_dist), key=lambda ii: loads[pe][ii])
            loads[pe][min_idx] += row_size

        max_bucket = max((loads[pe][ii] for pe in range(num_pes) for ii in range(ii_dist)), default=0)
        return int((max_bucket + padding) * ii_dist)

    run_len = 0
    tiles_out: list[dict] = []
    for tr in range(num_tiles_rows):
        for tc in range(num_tiles_cols):
            row_dict = tiles.get((tr, tc), {})
            if mode == "row_sharing":
                removed = shared_rows.get((tr, tc), [])
                tile_size = tile_runlen_row_sharing(row_dict, removed)
            else:
                tile_size = tile_runlen_no_rs(row_dict)
            run_len += tile_size
            tiles_out.append({"tr": tr, "tc": tc, "tile_size": tile_size})

    factor_bn = (n / n0) if n0 != 0 else 0.0
    t2_runlen = float(run_len) * factor_bn
    return {"run_len": run_len, "t2_runlen": t2_runlen, "tiles": tiles_out}


