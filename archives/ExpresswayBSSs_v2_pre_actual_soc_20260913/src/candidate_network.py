"""SOC-bin candidate networks, following the paper's four pruning steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .parameters import BusinessParameters, ENTRY_NODE, EXIT_NODE, NodeId

Arc = tuple[NodeId, NodeId]
SCHEMA_VERSION = 2
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "candidate_network.json"
_EPS = 1e-9


def enumerate_paths(arcs: Sequence[Arc], origin: NodeId = ENTRY_NODE, destination: NodeId = EXIT_NODE) -> list[tuple[Arc, ...]]:
    """Enumerate complete paths in stable input-arc order; reject cycles."""
    adjacent: dict[NodeId, list[NodeId]] = {}
    for source, target in arcs:
        if target not in adjacent.setdefault(source, []):
            adjacent[source].append(target)
    paths: list[tuple[Arc, ...]] = []

    def visit(node: NodeId, path: tuple[Arc, ...], visited: frozenset) -> None:
        if node == destination:
            paths.append(path)
            return
        for target in adjacent.get(node, []):
            if target in visited:
                raise ValueError("candidate network contains a cycle")
            visit(target, (*path, (node, target)), visited | {target})

    visit(origin, (), frozenset({origin}))
    return paths


def _arc_key(arc: Arc) -> str:
    return f"{arc[0]}->{arc[1]}"


def generate_candidate_network(params: BusinessParameters) -> dict:
    params.validate()
    od_networks = []
    for od_index, od in enumerate(params.od_pairs):
        nodes = params.od_nodes(od_index)
        positions = {node: params.node_position_km(od_index, node) for node in nodes}
        order = {node: index for index, node in enumerate(nodes)}
        bins = []
        for index, (low, high) in enumerate(params.soc_bins):
            raw: list[Arc] = []
            for source_index, source in enumerate(nodes[:-1]):
                for target in nodes[source_index + 1:]:
                    charge = high if source == ENTRY_NODE else 1.
                    need = params.min_exit_soc if target == EXIT_NODE else 0.
                    if charge - params.soc_consumption(od_index, source, target) >= need - _EPS:
                        raw.append((source, target))
            paths = enumerate_paths(raw)
            short_arcs = sorted(
                {arc for path in paths for arc in path if arc[1] != EXIT_NODE and positions[arc[1]] - positions[arc[0]] < params.min_swap_spacing_km - _EPS},
                key=lambda arc: (positions[arc[1]] - positions[arc[0]], order[arc[0]], order[arc[1]]),
            )
            # All later legs are already feasible from a full battery. The
            # first leg determines which complete paths the bin's lower SOC
            # can use, including the exit reserve for a direct-exit first leg.
            lower_paths = [path for path in paths
                           if low - params.soc_consumption(od_index, *path[0])
                           >= (params.min_exit_soc if path[0][1] == EXIT_NODE else 0.) - _EPS]
            removed = []
            # If the lower bound has no raw path, preserve every raw complete
            # path so feasible users higher in the bin are not pruned away.
            if lower_paths:
                for arc in short_arcs:
                    lower_alternatives = [path for path in lower_paths if arc not in path]
                    if lower_alternatives:
                        # Keep upper-SOC alternatives as well as the protected
                        # lower-SOC paths; do not replace the graph by the latter.
                        paths = [path for path in paths if arc not in path]
                        lower_paths = lower_alternatives
                        removed.append(arc)
            kept = {arc for path in paths for arc in path}
            candidate = [arc for arc in raw if arc in kept]
            bins.append({
                "soc_bin_index": index, "soc_lower": low, "soc_upper": high,
                "raw_arcs": [list(arc) for arc in raw],
                "removed_arcs": [list(arc) for arc in removed],
                "candidate_arcs": [list(arc) for arc in candidate],
                "arc_distance_km": {_arc_key(arc): params.distance_km(od_index, *arc) for arc in candidate},
                "arc_soc_consumption": {_arc_key(arc): params.soc_consumption(od_index, *arc) for arc in candidate},
                "complete_paths": [[list(arc) for arc in path] for path in paths],
            })
        od_networks.append({"od_index": od_index, "od_id": od.od_id, "nodes": nodes,
                            "node_positions_km": {str(key): value for key, value in positions.items()}, "soc_bins": bins})
    return {"schema_version": SCHEMA_VERSION, "soc_bins": [list(bounds) for bounds in params.soc_bins],
            "min_exit_soc": params.min_exit_soc, "min_swap_spacing_km": params.min_swap_spacing_km,
            "range_km": params.range_km, "od_networks": od_networks}


def _bin_record(network: dict, od_index: int, entry_soc: float) -> dict:
    bins = network["soc_bins"]
    index = next((h for h, (lo, hi) in enumerate(bins) if lo <= entry_soc < hi or h == len(bins) - 1 and lo <= entry_soc <= hi), None)
    if index is None:
        raise ValueError(f"entry_soc={entry_soc} is outside the network SOC bins")
    return network["od_networks"][od_index]["soc_bins"][index]


def get_candidate_arcs(network: dict, od_index: int, entry_soc: float) -> list[Arc]:
    """Return offline bin arcs before filtering the current first leg."""
    return [tuple(arc) for arc in _bin_record(network, od_index, entry_soc)["candidate_arcs"]]


def get_feasible_arcs(network: dict, od_index: int, entry_soc: float) -> list[Arc]:
    record = _bin_record(network, od_index, entry_soc)
    arcs = []
    for raw in record["candidate_arcs"]:
        arc = tuple(raw)
        if arc[0] == ENTRY_NODE:
            required = network["min_exit_soc"] if arc[1] == EXIT_NODE else 0.
            if entry_soc - record["arc_soc_consumption"][_arc_key(arc)] < required - _EPS:
                continue
        arcs.append(arc)
    # Direct exit never removes other complete reachable alternatives.
    complete = {arc for path in enumerate_paths(arcs) for arc in path}
    return [arc for arc in arcs if arc in complete]


def validate_candidate_network(network: dict, params: BusinessParameters | None = None) -> None:
    if network.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate network schema")
    if params is not None and network != generate_candidate_network(params):
        raise ValueError("candidate network does not match configured road and SOC bins")
    for od in network["od_networks"]:
        for record in od["soc_bins"]:
            arcs = [tuple(arc) for arc in record["candidate_arcs"]]
            if set(arcs) != {arc for path in enumerate_paths(arcs) for arc in path}:
                raise ValueError("candidate network contains arcs outside complete paths")


def save_candidate_network(network: dict, path: str | Path = DEFAULT_OUTPUT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidate_network(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    network = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_network(network)
    return network
