"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.signd.detector import Candidate
from openpilot.sunnypilot.selfdrive.signd.reader import ReaderParams, SignReader, Track
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection

DT = 0.05  # camera frames at 20 Hz
FRAME = (np.zeros(10, np.uint8), 8, 8, 8, 8)  # never looked at by the stub


def box4(box) -> tuple[int, int, int, int]:
  x0, y0, x1, y1 = (int(v) for v in box)
  return x0, y0, x1, y1


class StubDetector:
  """Records what the reader asks for. scan() finds the sign while `visible`; refine() finds it wherever it's asked to look."""
  def __init__(self, local=True, value: int | None = 50):
    self.local = object() if local else None
    self.value = value
    self.visible = True
    self.refine_ok = True
    self.calls: list[tuple[str, float]] = []
    self.t = 0.

  def scan(self, data, stride, uv_offset, w, h):
    self.calls.append(('scan', self.t))
    return [Candidate((100, 50, 120, 75), 0.9, self.value, 0.8)] if self.visible else []

  def refine(self, data, stride, uv_offset, w, h, box):
    self.calls.append(('read', self.t))
    if not self.refine_ok:
      return None
    return Candidate(box4(box), 0.9, self.value, 0.8)

  def to_detection(self, c, w, h):
    return None if c.value is None else Detection(c.value, c.confidence, (0., 0., 0.01, 0.02))


def run(reader, det, seconds, start=0.):
  out = []
  for i in range(round(seconds / DT)):
    det.t = start + i * DT
    out.append(reader.update(det.t, *FRAME))
  return out


class TestSignReader(OpenpilotTestCase):
  def test_scans_once_a_second_when_nothing_found(self):
    det = StubDetector()
    det.visible = False
    run(SignReader(det), det, 3.)
    assert [round(t, 2) for kind, t in det.calls] == [0., 1., 2.]
    assert all(kind == 'scan' for kind, _ in det.calls)

  def test_bursts_reads_between_scans_after_a_find(self):
    det = StubDetector()
    run(SignReader(det, ReaderParams(track_time=0.9)), det, 1.)
    reads = [t for kind, t in det.calls if kind == 'read']
    # 4 Hz after the scan at t=0, until the next scan
    assert [round(t, 2) for t in reads] == [0.25, 0.5, 0.75]

  def test_track_expires(self):
    det = StubDetector()
    run(SignReader(det, ReaderParams(scan_hz=0.1, track_time=1.)), det, 3.)
    reads = [t for kind, t in det.calls if kind == 'read']
    assert reads and max(reads) <= 1.
    assert len(det.calls) == 1 + len(reads)  # nothing after the track ended, until the next scan in 10 s

  def test_lost_sign_drops_track(self):
    det = StubDetector()
    det.refine_ok = False
    run(SignReader(det, ReaderParams(scan_hz=0.1, max_misses=2)), det, 3.)
    assert sum(kind == 'read' for kind, _ in det.calls) == 3  # misses 1, 2, 3 > max_misses

  def test_unreadable_sign_drops_track(self):
    # found every time but never readable (e.g. not a speed sign): must not hold the GPU for the whole track_time
    det = StubDetector(value=None)
    out = run(SignReader(det, ReaderParams(scan_hz=0.1, max_misses=2, track_time=5.)), det, 3.)
    assert sum(kind == 'read' for kind, _ in det.calls) == 3
    assert all(not d for d in out if d is not None)

  def test_readings_returned_only_when_work_ran(self):
    det = StubDetector()
    out = run(SignReader(det), det, 1.)
    ran = [i for i, d in enumerate(out) if d is not None]
    assert ran == [0, 5, 10, 15]
    assert all(d and d[0].value == 50 for d in out if d is not None)

  def test_round_robin_between_tracks(self):
    class TwoSigns(StubDetector):
      def __init__(self):
        super().__init__()
        self.looked: list[int] = []

      def scan(self, data, stride, uv_offset, w, h):
        return [Candidate((100, 50, 120, 75), 0.9, 50, 0.8), Candidate((300, 50, 320, 75), 0.8, 60, 0.7)]

      def refine(self, data, stride, uv_offset, w, h, box):
        self.looked.append(int(box[0]))
        return Candidate(box4(box), 0.9, 50, 0.8)

    det = TwoSigns()
    run(SignReader(det), det, 1.)
    assert det.looked == [100, 300, 100]

  def test_without_local_detector_scans_at_2hz(self):
    # no tracking possible: fixed full scans, as before tracking existed
    det = StubDetector(local=False)
    run(SignReader(det), det, 2.)
    assert [(kind, round(t, 2)) for kind, t in det.calls] == [('scan', 0.), ('scan', 0.5), ('scan', 1.), ('scan', 1.5)]


class TestTrack(OpenpilotTestCase):
  def test_predicts_motion_and_growth(self):
    tr = Track(box=np.array([100., 50., 120., 75.]), t=0., start_t=0.)
    tr.update(0.5, np.array([110., 45., 134., 75.]))  # moved right, grew 1.2x
    pred = tr.predict(1.0)
    h0, h1 = 25., 30.
    assert abs((pred[3] - pred[1]) - h1 * h1 / h0) < 0.5  # constant growth rate
    assert (pred[0] + pred[2]) / 2 > (110 + 134) / 2  # keeps moving right
