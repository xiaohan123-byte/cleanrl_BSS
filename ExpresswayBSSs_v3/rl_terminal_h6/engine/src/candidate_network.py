"""Shared full-battery road arcs and candidate paths for each user's SOC."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from .parameters import BusinessParameters, ENTRY_NODE, EXIT_NODE, NodeId

Arc = tuple[NodeId, NodeId]
SCHEMA_VERSION = 3
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


def _station_arcs(nodes, positions, range_km, min_exit_soc) -> list[Arc]:
    direction = 1 if positions[EXIT_NODE] > positions[ENTRY_NODE] else -1
    return [(source, target)
            for index, source in enumerate(nodes[1:-1], start=1)
            for target in nodes[index + 1:]
            if 1. - direction * (positions[target] - positions[source]) / range_km
            >= (min_exit_soc if target == EXIT_NODE else 0.) - _EPS]


def generate_candidate_network(params: BusinessParameters) -> dict:
    """Precompute unpruned arcs after a full-battery swap, once per OD."""
    params.validate()
    od_networks = []
    for od_index, od in enumerate(params.od_pairs):
        nodes = params.od_nodes(od_index)
        positions = {node: params.node_position_km(od_index, node) for node in nodes}
        arcs = _station_arcs(nodes, positions, params.range_km, params.min_exit_soc)
        od_networks.append({
            "od_index": od_index, "od_id": od.od_id, "nodes": nodes,
            "node_positions_km": {str(node): value for node, value in positions.items()},
            "station_arcs": [list(arc) for arc in arcs],
        })
    return {"schema_version": SCHEMA_VERSION,
            "min_exit_soc": params.min_exit_soc, "min_swap_spacing_km": params.min_swap_spacing_km,
            "range_km": params.range_km, "od_networks": od_networks}


def _od_record(network: dict, od_index: int) -> dict:
    if network.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate network schema; regenerate the network")
    if not 0 <= od_index < len(network["od_networks"]):
        raise ValueError("unknown OD index")
    return network["od_networks"][od_index]


def build_user_arcs(network: dict, od_index: int, origin: NodeId,
                    position_km: float, soc: float, *,
                    last_swap_position_km: float | None = None,
                    protected_stations: Sequence[int] | None = None) -> list[Arc]:
    """Connect the current origin using known SOC, then prune complete paths.

    A waiting user's origin is its station and SOC is the full battery it will
    receive; request precedence separately requires that swap before departure.
    A feasible existing station sequence is protected in its entirety.
    """
    record = _od_record(network, od_index)
    if not math.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError("user SOC must be finite and in [0, 1]")
    positions = {node: record["node_positions_km"][str(node)] for node in record["nodes"]}
    direction = 1 if positions[EXIT_NODE] > positions[ENTRY_NODE] else -1
    if (not math.isfinite(position_km)
            or direction * (position_km - positions[ENTRY_NODE]) < -_EPS
            or direction * (positions[EXIT_NODE] - position_km) <= 0):
        raise ValueError("user position must lie between the entry and exit")
    if origin in positions and abs(positions[origin] - position_km) > _EPS:
        raise ValueError("origin position does not match its road node")
    anchor = position_km if last_swap_position_km is None else last_swap_position_km
    if not math.isfinite(anchor) or direction * (anchor - position_km) > _EPS:
        raise ValueError("last swap position must not be downstream of the user")
    positions[origin] = position_km
    downstream = [node for node in record["nodes"][1:-1]
                  if direction * (positions[node] - position_km) > _EPS]
    nodes = [origin, *downstream, EXIT_NODE]
    order = {node: index for index, node in enumerate(nodes)}
    first = [(origin, node) for node in nodes[1:]
             if soc - direction * (positions[node] - position_km) / network["range_km"]
             >= (network["min_exit_soc"] if node == EXIT_NODE else 0.) - _EPS]
    downstream_set = set(downstream)
    arcs = first + [(a, b) for a, b in record["station_arcs"]
                    if a in downstream_set and (b in downstream_set or b == EXIT_NODE)]
    paths = enumerate_paths(arcs, origin)
    if not paths:
        return []
    protected = set()
    if protected_stations is not None:
        sequence = [origin, *protected_stations, EXIT_NODE]
        existing_path = tuple(zip(sequence, sequence[1:]))
        if existing_path in paths:
            protected.update(existing_path)

    def spacing(arc):
        # The virtual origin moves; the previous actual swap/entry does not.
        start = anchor if arc[0] == origin else positions[arc[0]]
        return direction * (positions[arc[1]] - start)

    short = sorted(
        {arc for path in paths for arc in path
         if arc[1] != EXIT_NODE and spacing(arc) < network["min_swap_spacing_km"] - _EPS},
        key=lambda arc: (spacing(arc), order[arc[0]], order[arc[1]]),
    )
    for arc in short:
        if arc in protected:
            continue
        alternatives = [path for path in paths if arc not in path]
        if alternatives:
            paths = alternatives
    kept = {arc for path in paths for arc in path}
    return [arc for arc in arcs if arc in kept]


def get_feasible_arcs(network: dict, od_index: int, entry_soc: float) -> list[Arc]:
    """Build an entrance network using the user's announced/measured SOC."""
    record = _od_record(network, od_index)
    return build_user_arcs(network, od_index, ENTRY_NODE,
                           record["node_positions_km"][ENTRY_NODE], entry_soc)


def validate_candidate_network(network: dict, params: BusinessParameters | None = None) -> None:
    if network.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate network schema; regenerate the network")
    for field, low, high in (("range_km", 0., math.inf),
                             ("min_exit_soc", 0., 1.), ("min_swap_spacing_km", 0., math.inf)):
        value = network[field]
        if not math.isfinite(value) or not low <= value <= high or (field == "range_km" and value == 0):
            raise ValueError(f"invalid candidate network {field}")
    for index, od in enumerate(network["od_networks"]):
        nodes = od["nodes"]
        if (od["od_index"] != index or len(nodes) < 2 or nodes[0] != ENTRY_NODE
                or nodes[-1] != EXIT_NODE or len(set(nodes)) != len(nodes)
                or any(not isinstance(node, int) for node in nodes[1:-1])):
            raise ValueError("invalid candidate network nodes")
        positions = {node: od["node_positions_km"][str(node)] for node in nodes}
        direction = 1 if positions[EXIT_NODE] > positions[ENTRY_NODE] else -1
        if (any(not math.isfinite(value) for value in positions.values())
                or any(direction * (positions[b] - positions[a]) <= 0 for a, b in zip(nodes, nodes[1:]))):
            raise ValueError("candidate nodes must be ordered downstream")
        expected = _station_arcs(nodes, positions, network["range_km"], network["min_exit_soc"])
        if od["station_arcs"] != [list(arc) for arc in expected]:
            raise ValueError("shared station arcs must match full-battery reachability")
    if params is not None and network != generate_candidate_network(params):
        raise ValueError("candidate network does not match configured road parameters")


def save_candidate_network(network: dict, path: str | Path = DEFAULT_OUTPUT_PATH) -> None:
    validate_candidate_network(network)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidate_network(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    network = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_network(network)
    return network
