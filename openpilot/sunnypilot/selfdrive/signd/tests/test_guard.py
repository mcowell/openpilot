"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.signd.guard import GuardLevel, ModelGuard

DT = 0.05  # modeld runs at 20 Hz


def frame_drop_perc(seconds: float, drops_at: tuple[float, ...] = ()):
  """modelV2.frameDropPerc as modeld computes it (modeld.py), for frames dropped at the given times"""
  f = FirstOrderFilter(0., 10., DT)
  drops = {round(t / DT) for t in drops_at}
  for i in range(round(seconds / DT)):
    x = f.update(1. if i in drops else 0.)
    yield i * DT, 100. * x / (1. + x)


def run(guard: ModelGuard, seconds: float, drops_at: tuple[float, ...] = ()) -> list[tuple[float, GuardLevel]]:
  return [(t, guard.update(t, perc)) for t, perc in frame_drop_perc(seconds, drops_at)]


def level_at(levels, t):
  return next(lv for tt, lv in levels if tt >= t - 1e-6)


class TestModelGuard(OpenpilotTestCase):
  def test_one_dropped_frame_is_near_openpilots_limit(self):
    # why the guard reacts to the first drop: selfdrived alerts above 1%
    peak = max(p for _, p in frame_drop_perc(5., drops_at=(1.,)))
    assert 0.4 < peak < 0.6
    peak2 = max(p for _, p in frame_drop_perc(5., drops_at=(1., 1.5)))
    assert peak2 > 0.9

  def test_normal_without_drops(self):
    assert all(lv == GuardLevel.normal for _, lv in run(ModelGuard(), 60.))

  def test_drop_pauses_then_reduces_then_recovers(self):
    levels = run(ModelGuard(), 40., drops_at=(1.,))
    assert level_at(levels, 0.5) == GuardLevel.normal
    assert level_at(levels, 1.) == GuardLevel.paused
    assert level_at(levels, 5.9) == GuardLevel.paused
    assert level_at(levels, 6.1) == GuardLevel.reduced
    # the filtered value is still decaying here, but that's not a new drop
    assert level_at(levels, 25.) == GuardLevel.reduced
    assert level_at(levels, 26.2) == GuardLevel.normal

  def test_drop_while_recovering_pauses_longer(self):
    levels = run(ModelGuard(), 40., drops_at=(1., 10.))
    assert level_at(levels, 10.) == GuardLevel.paused
    assert level_at(levels, 19.9) == GuardLevel.paused  # 10 s: doubled
    assert level_at(levels, 20.1) == GuardLevel.reduced

  def test_pause_is_capped(self):
    g = ModelGuard()
    t = 0.
    for _ in range(10):  # a drop each time it comes out of a pause
      while g.update(t, 0.) == GuardLevel.paused:
        t += DT
      g.update(t, 0.5)
      g._prev = 0.  # next drop is new again
      t += DT
    assert g._pause == g.params.max_pause_time
