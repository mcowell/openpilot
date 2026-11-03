"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Backs signd off when the driving model drops frames, so signd can never be what pushes openpilot over its limit.
selfdrived raises modeldLagging (a soft disable) when modelV2.frameDropPerc > 1. modeld filters that value with a
10 s time constant, so a single dropped frame reads ~0.5% and two close together ~1%: the guard reacts to the first.
It watches for new drops (the filtered value jumping up), not the value itself, which takes ~15 s to decay after one.
"""
from dataclasses import dataclass
from enum import IntEnum


class GuardLevel(IntEnum):
  normal = 0  # full schedule: scans and burst reads
  reduced = 1  # scans only, at the reader's scan rate
  paused = 2  # no detector work


@dataclass
class GuardParams:
  drop_step: float = 0.05  # % rise in frameDropPerc between updates that means a new dropped frame (one adds ~0.5)
  pause_time: float = 5.  # s, first pause; doubles each time drops recur while recovering
  max_pause_time: float = 60.
  recover_time: float = 20.  # s without drops in reduced before going back to normal


class ModelGuard:
  def __init__(self, params: GuardParams | None = None):
    self.params = params or GuardParams()
    self.level = GuardLevel.normal
    self._pause = self.params.pause_time
    self._until = 0.  # end of the current pause
    self._clear_since: float | None = None
    self._prev = 0.
    self.pauses = 0

  def update(self, t: float, frame_drop_perc: float) -> GuardLevel:
    p = self.params
    dropping = frame_drop_perc - self._prev > p.drop_step
    self._prev = frame_drop_perc

    if self.level == GuardLevel.paused:
      if t >= self._until:
        self.level, self._clear_since = GuardLevel.reduced, None
      return self.level

    if dropping:
      if self.level == GuardLevel.reduced:
        self._pause = min(self._pause * 2, p.max_pause_time)  # recurring while recovering: back off longer
      self.level, self._until = GuardLevel.paused, t + self._pause
      self.pauses += 1
      return self.level

    if self.level == GuardLevel.reduced:
      if self._clear_since is None:
        self._clear_since = t
      if t - self._clear_since >= p.recover_time:
        self.level, self._pause = GuardLevel.normal, p.pause_time
    return self.level
