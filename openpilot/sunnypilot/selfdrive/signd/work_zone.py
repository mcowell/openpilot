"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Work zone state from the reader's work zone checks (road work signs: Ontario TC-2, the orange diamond with a person
digging, and flagger signs). Ontario posts no end of work zone sign, so the zone ends hold_distance after the last road
work sign. Independent of the speed limit: a work zone often has no posted limit of its own.
"""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class WorkZoneParams:
  min_prob: float = 0.7  # per check
  min_hits: int = 2  # checks >= min_prob within window
  window: float = 3.  # s
  hold_distance: float = 2500.  # m after the last road work sign


@dataclass
class WorkZoneState:
  active: bool = False
  distance_since_sign: float = 0.  # m since the last road work sign (while active)


@dataclass
class WorkZoneFilter:
  params: WorkZoneParams = field(default_factory=WorkZoneParams)

  def __post_init__(self):
    self._hits: deque[float] = deque()
    self._last_t: float | None = None
    self.state = WorkZoneState()

  def reset(self) -> None:
    self._hits.clear()
    self.state = WorkZoneState()

  def update(self, t: float, reads: list[tuple[float, float]], v_ego: float | None = None) -> WorkZoneState:
    """reads: (t, probability) of work zone checks since the last update. v_ego None: unknown, the zone doesn't expire."""
    p = self.params
    dt = 0. if self._last_t is None else max(0., t - self._last_t)
    self._last_t = t
    if self.state.active:
      self.state.distance_since_sign += max(0., v_ego or 0.) * dt
      if self.state.distance_since_sign > p.hold_distance:
        self.state = WorkZoneState()

    new = [rt for rt, prob in reads if prob >= p.min_prob]
    self._hits.extend(new)
    while self._hits and t - self._hits[0] > p.window:
      self._hits.popleft()
    if new and len(self._hits) >= p.min_hits:
      self.state = WorkZoneState(active=True)  # (re)starts the hold distance
    return self.state
