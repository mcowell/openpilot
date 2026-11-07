"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.signd.detector import Candidate, Window
from openpilot.sunnypilot.selfdrive.signd.guard import GuardLevel
from openpilot.sunnypilot.selfdrive.signd.reader import ReaderParams, SignReader, Track
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection

DT = 0.05  # camera / driving model frames at 20 Hz
FRAME = (np.zeros(10, np.uint8), 8, 8, 8, 8)  # never looked at by the stub
SIGN = (100., 50., 120., 75.)
TILES = [(0, 0, 200, 200), (200, 0, 400, 200), (0, 200, 200, 400), (200, 200, 400, 400)]


def window(rect) -> Window:
  return Window(np.zeros(1), 1., 0, 0, rect[0], rect[1])


class StubDetector:
  """The sign is in tile 0 while `visible`. Local windows find it while `local_ok`. Records each GPU run."""
  def __init__(self, local=True, value: int | None = 50):
    self.local = object() if local else None
    self.value = value
    self.visible = True
    self.local_ok = True
    self.runs: list[str] = []

  def tiles(self, w, h):
    return TILES

  def entry_tiles(self, w, h):
    return TILES[:1]

  def prepare_tile(self, data, stride, uv_offset, w, h, rect):
    return window(rect)

  def prepare_local(self, data, stride, uv_offset, w, h, box):
    return window((-1, -1))  # marks a local window

  def run_window(self, win, w, h):
    is_local = win.x0 == -1
    self.runs.append('local' if is_local else f'tile{TILES.index((win.x0, win.y0, win.x0 + 200, win.y0 + 200))}')
    found = (is_local and self.local_ok) or (not is_local and self.visible and (win.x0, win.y0) == (0, 0))
    return np.array([[*SIGN, 0.9]]) if found else np.zeros((0, 5))

  def prepare_crop(self, data, stride, uv_offset, w, h, box):
    return np.zeros(1)

  def run_classifier(self, x):
    self.runs.append('classify')
    return self.value, 0.9

  def candidate(self, box, score, value, prob):
    x0, y0, x1, y1 = (int(v) for v in box)
    return Candidate((x0, y0, x1, y1), score, value, score * prob if value is not None else 0.)

  def scan(self, data, stride, uv_offset, w, h):
    self.runs.append('scan')
    return [Candidate((100, 50, 120, 75), 0.9, self.value, 0.8)] if self.visible else []

  def to_detection(self, c, w, h):
    return None if c.value is None else Detection(c.value, c.confidence, (0., 0., 0.01, 0.02))


def run(reader, seconds, level=GuardLevel.normal):
  return [reader.update(i * DT, *FRAME, level) for i in range(round(seconds / DT))]


class TestSignReader(OpenpilotTestCase):
  def test_sweeps_tiles_when_nothing_found(self):
    det = StubDetector()
    det.visible = False
    out = run(SignReader(det), 0.8)
    # one job every 2nd frame, entry tiles (here tile 0) alternating with the full grid
    assert det.runs == ['tile0', 'tile0', 'tile0', 'tile1', 'tile0', 'tile2', 'tile0', 'tile3']
    assert [i for i, d in enumerate(out) if d is not None] == list(range(0, 16, 2))

  def test_found_sign_is_classified_then_read(self):
    det = StubDetector()
    out = run(SignReader(det, ReaderParams(track_time=10.)), 1.)
    # found in tile 0; while tracking a job runs every frame: a local read (re-finds it precisely) and its classify, back to
    # back for the first new_track_time, then every 1/5 s with tiles in the free frames
    assert det.runs[:16] == ['tile0', 'local', 'classify', 'local', 'classify', 'tile0', 'local', 'classify', 'tile0',
                             'local', 'classify', 'local', 'classify', 'tile1', 'tile0', 'tile2']
    assert det.runs.count('local') == 6 and det.runs.count('classify') == 6
    # only classify jobs produce readings
    readings = [d for d in out if d]
    assert len(readings) == 6 and all(d[0].value == 50 for d in readings)

  def test_new_sign_read_twice_quickly(self):
    # a sign seen while turning can be in view for only ~0.3 s, and the filter needs two readings
    det = StubDetector()
    out = run(SignReader(det), 0.3)
    assert len([d for d in out if d]) >= 2

  def test_every_job_is_small(self):
    det = StubDetector()
    run(SignReader(det), 3.)
    assert 'scan' not in det.runs  # the long full-ROI scan only runs without a local detector

  def test_paused_does_nothing(self):
    det = StubDetector()
    assert all(d is None for d in run(SignReader(det), 2., level=GuardLevel.paused))
    assert det.runs == []

  def test_reduced_halves_jobs_and_doesnt_track(self):
    det = StubDetector()
    run(SignReader(det), 1., level=GuardLevel.reduced)
    assert 'local' not in det.runs
    assert det.runs == ['tile0', 'tile0', 'tile0', 'tile1', 'tile0']  # one job every 4th frame, tiles only

  def test_lost_sign_drops_track(self):
    det = StubDetector()
    det.local_ok = False
    det.visible = False
    reader = SignReader(det, ReaderParams(track_time=10.))
    reader.tracks.append(Track(box=np.array(SIGN), t=0., start_t=0.))
    run(reader, 3.)
    assert det.runs.count('local') == 3 and not reader.tracks  # misses 1, 2, 3 > max_misses

  def test_unreadable_sign_drops_track(self):
    # found every time but never readable (e.g. a licence plate): must not keep the GPU busy for the whole track_time
    det = StubDetector(value=None)
    reader = SignReader(det, ReaderParams(track_time=10.))
    out = run(reader, 2.)
    assert all(not d for d in out if d is not None)
    assert det.runs.count('local') <= 3

  def test_classify_queue_is_bounded(self):
    det = StubDetector()
    reader = SignReader(det, ReaderParams(max_pending=2))
    job = reader.prepare(0., *FRAME)
    assert job is not None
    for _ in range(5):
      reader._queue(job, np.array(SIGN), 0.9, Track(np.array(SIGN), 0., 0.))
    assert len(reader.pending) == 2

  def test_without_local_detector_scans_at_2hz(self):
    det = StubDetector(local=False)
    run(SignReader(det), 2.)
    assert det.runs == ['scan'] * 4

  def test_reduced_without_local_detector_scans_at_1hz(self):
    det = StubDetector(local=False)
    run(SignReader(det), 2., level=GuardLevel.reduced)
    assert det.runs == ['scan'] * 2


class TestTrack(OpenpilotTestCase):
  def test_predicts_motion_and_growth(self):
    tr = Track(box=np.array([100., 50., 120., 75.]), t=0., start_t=0.)
    tr.update(0.5, np.array([110., 45., 134., 75.]))  # moved right, grew 1.2x
    pred = tr.predict(1.0)
    h0, h1 = 25., 30.
    assert abs((pred[3] - pred[1]) - h1 * h1 / h0) < 0.5  # constant growth rate
    assert (pred[0] + pred[2]) / 2 > (110 + 134) / 2  # keeps moving right
