"""Typed state and interfaces for discrete MPC. Hours, kW and kWh throughout."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

UserKey = tuple[int, int]
NodeId = int | str
Arc = tuple[NodeId, NodeId]

class DomainError(ValueError):
    pass

def user_key_text(key) -> str:
    return f"{int(key[0])}:{int(key[1])}"

def _soc(value, label, returned=False):
    if not isfinite(value) or not 0 <= value <= 1 or (returned and value == 1):
        raise DomainError(f"{label} must be in {'[0, 1)' if returned else '[0, 1]'}")

@dataclass
class WaitingRequest:
    request_id: str
    station: int
    kind: str
    arrival_time: float
    deadline: float
    return_soc: float
    user_key: UserKey | None = None

    def __post_init__(self):
        if self.kind not in {'reservation', 'random'}:
            raise DomainError('unknown request kind')
        if not self.request_id or not isfinite(self.arrival_time):
            raise DomainError('request needs ID and finite arrival time')
        if not isfinite(self.deadline) or self.deadline < self.arrival_time:
            raise DomainError('deadline precedes arrival')
        _soc(self.return_soc, 'return_soc', returned=True)
        if self.user_key is not None:
            self.user_key = tuple(self.user_key)
        if self.kind == 'reservation' and self.user_key is None:
            raise DomainError('reservation request needs user key')

    def to_dict(self):
        return asdict(self)

@dataclass
class Reservation:
    user_key: UserKey
    entry_time: float
    entry_soc: float
    position_km: float
    soc: float
    entered: bool = False
    day_ahead: list[int] = field(default_factory=list)
    retained_plan: list[int] = field(default_factory=list)
    published_plan: list[int] | None = None
    last_swap_position_km: float = 0.0
    waiting_request_id: str | None = None
    next_arrival_time: float | None = None
    completed_stations: list[int] = field(default_factory=list)
    status: str = 'active'
    actual_entry_time: float | None = None
    observation_time: float | None = None
    observed_segment_index: int | None = None
    observed_speed_kmh: float | None = None

    def __post_init__(self):
        self.user_key = tuple(self.user_key)
        _soc(self.entry_soc, 'entry_soc')
        _soc(self.soc, 'vehicle SOC')
        if not isfinite(self.entry_time) or self.entry_time < 0:
            raise DomainError('entry time must be finite and nonnegative')
        if not isfinite(self.position_km) or not isfinite(self.last_swap_position_km):
            raise DomainError('vehicle positions must be finite')
        if self.next_arrival_time is not None and not isfinite(self.next_arrival_time):
            raise DomainError('next arrival time must be finite when known')
        if self.actual_entry_time is not None and (not self.entered or not isfinite(self.actual_entry_time) or self.actual_entry_time < 0):
            raise DomainError('actual entry time is only visible after entry')
        if self.observation_time is not None and (not isfinite(self.observation_time) or self.observation_time < 0):
            raise DomainError('observation time must be finite and nonnegative')
        if self.observed_speed_kmh is not None and (not isfinite(self.observed_speed_kmh) or self.observed_speed_kmh <= 0):
            raise DomainError('observed road speed must be finite and positive')
        if self.observed_segment_index is not None and (not isinstance(self.observed_segment_index, int) or self.observed_segment_index < 0):
            raise DomainError('observed physical segment index must be nonnegative')
        if self.status not in {'active', 'completed', 'failed'}:
            raise DomainError('unknown reservation status')

    def to_dict(self):
        return asdict(self)

@dataclass
class RollingState:
    period: int
    slot_soc: list[list[float]]
    users: dict[str, Reservation] = field(default_factory=dict)
    waiting: dict[str, WaitingRequest] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)
    seen_random_ids: list[str] = field(default_factory=list)

    def clone(self):
        return deepcopy(self)

    def validate(self):
        if not isinstance(self.period, int) or self.period < 0:
            raise DomainError('period must be nonnegative integer')
        for row in self.slot_soc:
            for value in row:
                _soc(value, 'slot SOC')
        for key, user in self.users.items():
            user.__post_init__()
            if key != user_key_text(user.user_key):
                raise DomainError('inconsistent user key')
            if user.waiting_request_id is not None:
                request = self.waiting.get(user.waiting_request_id)
                if request is None or request.user_key != user.user_key:
                    raise DomainError('inconsistent user waiting request')
        for key, request in self.waiting.items():
            request.__post_init__()
            if key != request.request_id:
                raise DomainError('inconsistent request ID')
            if request.kind == 'reservation':
                user = self.users.get(user_key_text(request.user_key))
                if user is None or user.waiting_request_id != key or user.status != 'active':
                    raise DomainError('orphaned reservation waiting request')
        ids = [event['event_id'] for event in self.ledger]
        if len(ids) != len(set(ids)):
            raise DomainError('duplicate ledger event ID')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        value = deepcopy(payload)
        value['users'] = {key: Reservation(**record) for key, record in value.get('users', {}).items()}
        value['waiting'] = {key: WaitingRequest(**record) for key, record in value.get('waiting', {}).items()}
        result = cls(**value)
        result.validate()
        return result

@dataclass
class CandidateRequest:
    request_id: str
    station: int
    kind: str
    return_soc: float
    user_key: UserKey | None = None
    arc: Arc | None = None
    predecessors: tuple[str, ...] = ()
    arrival_time: float | None = None
    travel_time: float = 0.0
    observed: bool = False
    deadline: float | None = None

    def __post_init__(self):
        _soc(self.return_soc, 'candidate return_soc', returned=True)
        if self.user_key is not None:
            self.user_key = tuple(self.user_key)
        if self.arc is not None:
            self.arc = tuple(self.arc)
        self.predecessors = tuple(self.predecessors)
        if self.predecessors and (not isfinite(self.travel_time) or self.travel_time <= 0):
            raise DomainError('dependent request needs positive travel time')
        if not self.predecessors and self.arrival_time is None:
            raise DomainError('independent request needs arrival time')

@dataclass
class UserNetwork:
    user_key: UserKey
    origin: NodeId
    arcs: list[Arc]
    reference_stations: tuple[int, ...]
    frozen_stations: tuple[int, ...]
    can_update: bool

@dataclass
class MPCWindow:
    ell: int
    horizon: int
    state: RollingState
    networks: dict[str, UserNetwork]
    requests: list[CandidateRequest]

@dataclass
class ServiceDecision:
    request_id: str
    station: int
    slot: int
    period: int

@dataclass
class MPCSolution:
    status: str
    objective: float
    objective_terms: dict[str, float]
    paths: dict[str, list[int]]
    services: list[ServiceDecision]
    power: list[list[list[float]]]
    soc: list[list[list[float]]]
    request_outcomes: dict[str, dict] = field(default_factory=dict)
    mip_gap: float = 0.0
    solve_seconds: float = 0.0
    terminal_features: list[float] = field(default_factory=list)
    terminal_value: float = 0.0
    build_seconds: float = 0.0
    wall_seconds: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

@dataclass
class ExecutionResult:
    state: RollingState
    events: list[dict]
    reward: float

    def to_dict(self):
        return {'state': self.state.to_dict(), 'events': deepcopy(self.events), 'reward': self.reward}

def initial_state(params: Any, reservations: list[dict], plans: dict[str, list[int]]) -> RollingState:
    users = {}
    for record in reservations:
        key = tuple(record['user_key'])
        text = user_key_text(key)
        if text in users:
            raise DomainError(f'duplicate user {text}')
        od = next(od for od in params.od_pairs if od.od_id == key[0])
        plan = list(plans[text])
        entry_time = float(record['entry_time'])
        entered = entry_time <= 0.0 and not getattr(params, 'terminal_experiment', False)
        users[text] = Reservation(
            user_key=key, entry_time=entry_time, entry_soc=float(record['entry_soc']),
            position_km=od.entry_km, soc=float(record['entry_soc']), entered=entered,
            day_ahead=plan.copy(), retained_plan=plan.copy(),
            published_plan=plan.copy() if entered else None,
            last_swap_position_km=od.entry_km, observation_time=0.,
            actual_entry_time=entry_time if entered else None)
    result = RollingState(0, deepcopy(params.station.initial_slot_soc), users)
    result.validate()
    return result
