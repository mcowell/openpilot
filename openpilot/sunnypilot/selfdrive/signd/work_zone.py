"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Work zone state from the reader's work zone checks (road work signs: Ontario TC-2, the orange diamond with a person
digging, and flagger signs). Ontario posts no end of work zone sign, so the zone ends hold_distance after the last road
work sign, or sooner when the car leaves the road the sign was on: another road name for road_change_distance, or no
road name after a sharp turn (into a parking lot). Independent of the speed limit: a work zone often has no posted limit
of its own.
"""
import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class WorkZoneParams:
  min_prob: float = 0.7  # per check
  min_hits: int = 2  # checks >= min_prob within window
  window: float = 3.  # s
  hold_distance: float = 2500.  # m after the last road work sign
  # leaving the sign's road (as SignFilter does for the limit)
  road_change_distance: float = 100.  # m on another road name: map matching flickers at intersections
  road_grace_distance: float = 200.  # m; a sign read this soon before the name changes, with no turn since, is on the new road
  off_road_distance: float = 15.  # m with no road name, after a sharp turn
  turn_window: float = 10.  # s
  turn_reset_angle: float = math.radians(60.)
  grace_max_turn: float = math.radians(30.)  # more turning than this since the sign: it was on the old road


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
    self._yaw: deque[tuple[float, float]] = deque()  # (t, heading change during step)
    self._clear_road()
    self.state = WorkZoneState()

  def _clear_road(self) -> None:
    self._road = ''  # road of the last road work sign, '' if not known yet
    self._new_road, self._new_road_dist, self._off_road_dist = '', 0., 0.
    self._turn_since_sign = 0.  # largest heading change (over turn_window) since the last sign

  def reset(self) -> None:
    self._hits.clear()
    self._yaw.clear()
    self._clear_road()
    self.state = WorkZoneState()

  def update(self, t: float, reads: list[tuple[float, float]], v_ego: float | None = None, road: str = '',
             yaw_rate: float = 0.) -> WorkZoneState:
    """reads: (t, probability) of work zone checks since the last update. v_ego None: unknown, the zone doesn't expire.
    road: road name from the map, '' if not known right now."""
    p = self.params
    dt = 0. if self._last_t is None else max(0., t - self._last_t)
    self._last_t = t
    self._yaw.append((t, yaw_rate * dt))
    while self._yaw and t - self._yaw[0][0] > p.turn_window:
      self._yaw.popleft()
    if self.state.active and self._left_road(max(0., v_ego or 0.) * dt, road):
      self.reset()

    new = [rt for rt, prob in reads if prob >= p.min_prob]
    self._hits.extend(new)
    while self._hits and t - self._hits[0] > p.window:
      self._hits.popleft()
    if new and len(self._hits) >= p.min_hits:
      self.state = WorkZoneState(active=True)  # (re)starts the hold distance
      self._clear_road()
      self._road = road
      self._yaw.clear()
    return self.state

  def _left_road(self, step: float, road: str) -> bool:
    p = self.params
    self.state.distance_since_sign += step
    if self.state.distance_since_sign > p.hold_distance:
      return True
    heading = abs(sum(d for _, d in self._yaw))
    self._turn_since_sign = max(self._turn_since_sign, heading)
    turned = heading > p.turn_reset_angle

    if not self._road:
      self._road = road  # the sign was read before the map knew the road
      return False
    if not road:
      self._off_road_dist += step
      return turned and self._off_road_dist >= p.off_road_distance
    self._off_road_dist = 0.
    if road == self._road:
      self._new_road, self._new_road_dist = '', 0.
      return False
    if road != self._new_road:
      self._new_road, self._new_road_dist = road, 0.
    self._new_road_dist += step
    if turned or self._new_road_dist >= p.road_change_distance:
      if self.state.distance_since_sign > p.road_grace_distance or turned or self._turn_since_sign > p.grace_max_turn:
        return True
      self._road, self._new_road, self._new_road_dist = road, '', 0.  # read just before the name caught up
    return False
