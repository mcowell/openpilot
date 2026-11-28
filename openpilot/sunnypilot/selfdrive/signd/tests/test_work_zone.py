"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.sunnypilot.selfdrive.signd.work_zone import WorkZoneFilter, WorkZoneParams

SIGN = {0.2: 0.9, 0.4: 0.9}  # two road work sign checks: the zone starts at 0.4 s


def drive(f, t0, t1, v=15., reads=None, road='', yaw_deg=0.):
  """update at 20 Hz from t0 to t1; reads: {t: probability} handed over at that update; yaw_deg: deg/s"""
  reads = reads or {}
  t = t0
  while t < t1 - 1e-9:
    s = f.update(t, [(t, reads[round(t, 2)])] if round(t, 2) in reads else [], v, road, math.radians(yaw_deg))
    t = round(t + 0.05, 2)
  return s


class TestWorkZone:
  def test_two_readings_start_the_zone(self):
    f = WorkZoneFilter()
    assert not drive(f, 0., 1., reads={0.5: 0.9}).active  # one reading isn't enough
    assert drive(f, 1., 2., reads={1.5: 0.8}).active  # a second within 3 s

  def test_readings_too_far_apart_or_too_weak(self):
    f = WorkZoneFilter()
    assert not drive(f, 0., 6., reads={0.5: 0.9, 4.: 0.9}).active
    f = WorkZoneFilter()
    assert not drive(f, 0., 2., reads={0.5: 0.6, 1.: 0.65, 1.5: 0.69}).active

  def test_ends_after_hold_distance_from_the_last_sign(self):
    f = WorkZoneFilter(WorkZoneParams(hold_distance=1000.))
    assert drive(f, 0., 1., reads={0.2: 0.9, 0.4: 0.9}).active
    assert drive(f, 1., 60., v=15.).active  # ~890 m
    # another road work sign restarts the distance
    assert drive(f, 60., 61., reads={60.2: 0.9, 60.4: 0.9}).active
    assert drive(f, 61., 120., v=15.).active
    assert not drive(f, 120., 130., v=15.).active  # > 1000 m since the last sign

  def test_stopped_in_the_zone_keeps_it(self):
    f = WorkZoneFilter(WorkZoneParams(hold_distance=100.))
    drive(f, 0., 1., reads={0.2: 0.9, 0.4: 0.9})
    assert drive(f, 1., 600., v=0.).active

  def test_turn_onto_another_road_ends_the_zone(self):
    f = WorkZoneFilter()
    assert drive(f, 0., 10., reads=SIGN, road='Centrepointe Drive').active
    assert drive(f, 10., 14., v=6., yaw_deg=25., road='Centrepointe Drive').active  # 100 deg, name not updated yet
    assert not drive(f, 14., 14.5, v=6., road='Baseline Road').active

  def test_turn_into_parking_lot_ends_the_zone(self):
    # Centrepointe Drive, 2026-11-27: the zone stayed on in the lot
    f = WorkZoneFilter()
    drive(f, 0., 10., reads=SIGN, road='Centrepointe Drive')
    drive(f, 10., 13., v=5., yaw_deg=-30., road='')
    assert not drive(f, 13., 16., v=5., road='').active

  def test_stays_on_along_the_road(self):
    f = WorkZoneFilter()
    drive(f, 0., 1., reads=SIGN, road='Greenbank Road')
    assert drive(f, 1., 3., v=8., yaw_deg=20., road='Greenbank Road').active  # a 40 deg bend
    assert drive(f, 3., 15., road='Greenbank Road').active
    assert drive(f, 15., 25., road='').active  # map name dropout without a turn
    assert drive(f, 15., 30., v=0., road='Fallowfield Road').active  # stopped on a cross street's name
    assert drive(f, 30., 100., road='Greenbank Road').active

  def test_other_road_name_for_a_while_ends_the_zone(self):
    f = WorkZoneFilter()
    drive(f, 0., 1., reads=SIGN, road='Greenbank Road')
    drive(f, 1., 20., road='Greenbank Road')  # 285 m
    assert drive(f, 20., 25., road='Greenbank Road Ramp').active  # 75 m: could be map matching at an intersection
    assert not drive(f, 25., 30., road='Greenbank Road Ramp').active

  def test_sign_just_before_the_name_changes_belongs_to_the_new_road(self):
    f = WorkZoneFilter()
    drive(f, 0., 1., v=10., reads=SIGN, road='Baseline Road')  # read just after turning, name not updated yet
    assert drive(f, 1., 20., v=10., road='Southwood Drive').active
    assert drive(f, 20., 60., v=10., road='Southwood Drive').active
