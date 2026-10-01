"""Reachable minimum-swap reference paths, independent of station inventory."""

from __future__ import annotations

from .candidate_network import enumerate_paths, get_feasible_arcs
from .parameters import BusinessParameters


def user_key_text(user_key) -> str:
    return f"{int(user_key[0])}:{int(user_key[1])}"


def generate_dayahead_plan(params: BusinessParameters, network: dict, reservations: list[dict]) -> dict[str, list[int]]:
    """Minimize swaps; ties prefer downstream stations lexicographically."""
    result = {}
    for reservation in reservations:
        key = user_key_text(reservation["user_key"])
        if key in result:
            raise ValueError(f"duplicate reservation {key}")
        od_index = params.od_index(reservation["od_id"])
        reach_soc = float(reservation["entry_soc"])
        if getattr(params, "terminal_experiment", False):
            reach_soc = max(.5, reach_soc - params.entry_soc_error)
        arcs = get_feasible_arcs(network, od_index, reach_soc)
        paths = enumerate_paths(arcs)
        if not paths:
            raise ValueError(f"reservation {key} has no complete reachable candidate path")
        sequences = [[int(target) for _, target in path if isinstance(target, int)] for path in paths]
        od = params.od_pairs[od_index]
        direction = 1 if od.exit_km > od.entry_km else -1
        result[key] = min(sequences, key=lambda seq: (len(seq), tuple(-direction * params.station.positions_km[i] for i in seq)))
    return result
