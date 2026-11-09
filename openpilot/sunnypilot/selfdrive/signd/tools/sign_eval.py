#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Offline evaluation of vision speed limit detection on recorded drives, no car needed.

  # run detector + filter over a route, write a timeline and an annotated video
  sign_eval.py ~/drives/2026-11-01--10-00-00 --out out/ --video
  # score against hand labels
  sign_eval.py ~/drives/2026-11-01--10-00-00 --labels labels.csv
  # no model yet: dump ROI crops at 1 Hz to build a training/labeling set
  sign_eval.py ~/drives/2026-11-01--10-00-00 --dump-rois rois/ --rate 1

ROUTE can be a single segment directory (<route>--<n>, as in /data/media/0/realdata) or a
directory containing segment directories. Frames come from fcamera.hevc (--camera road, default)
or ecamera.hevc (--camera wide). If the openpilot python env is available, vEgo and yaw rate are
read from rlog/qlog so distance and turn resets in the filter behave as on the car.

labels.csv columns: route,segment,start_frame,end_frame,limit
  frames are 20 Hz camera frame indices within the segment; limit is the posted limit in effect
  (0 = no posted limit known). Time outside labeled intervals is not scored.
"""
import argparse
import csv
import json
from bisect import bisect_right
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection, SignFilter

CAMERA_FPS = 20
SEGMENT_SECONDS = 60
CAMERA_FILES = {'wide': ('ecamera.hevc',), 'road': ('fcamera.hevc',)}
SEGMENT_RE = re.compile(r'^(?P<route>.+)--(?P<seg>\d+)$')


def find_segments(path: Path) -> list[tuple[str, int, Path]]:
  m = SEGMENT_RE.match(path.name)
  if m and any(path.glob('*camera*')):
    return [(m['route'], int(m['seg']), path)]
  segs = []
  for p in path.iterdir():
    m = SEGMENT_RE.match(p.name)
    if p.is_dir() and m:
      segs.append((m['route'], int(m['seg']), p))
  return sorted(segs, key=lambda s: (s[0], s[1]))


def probe_size(fn: Path) -> tuple[int, int]:
  out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                 '-of', 'csv=p=0', str(fn)], text=True)
  w, h = out.strip().split(',')[:2]
  return int(w), int(h)


def iter_frames(fn: Path, rate: float):
  """Yields (camera frame index, NV12 frame) sampled at `rate` Hz. NV12 is the camera's VisionIPC format, so frames take
  exactly the on-device path (stride = width, chroma right after luma)."""
  w, h = probe_size(fn)
  step = max(1, round(CAMERA_FPS / rate))
  # decode every frame and subsample here: raw hevc has no timestamps, and the minimal ffmpeg
  # build in the openpilot venv has no filters
  proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-vsync', '0', '-i', str(fn), '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1'],
                          stdout=subprocess.PIPE)
  assert proc.stdout is not None
  size = w * h * 3 // 2
  frame = 0
  while True:
    buf = proc.stdout.read(size)
    if len(buf) < size:
      break
    if frame % step == 0:
      yield frame, np.frombuffer(buf, dtype=np.uint8)
    frame += 1
  proc.kill()
  proc.wait()


def road_at(roads: list[tuple[float, str]], t: float) -> str:
  """Latest road name logged at or before t, '' if none yet"""
  i = bisect_right(roads, t, key=lambda r: r[0])
  return roads[i - 1][1] if i else ''


def load_vehicle_state(seg_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, str]]] | None:
  """(t, vEgo, yawRate, [(t, road name)]) from the segment log, t in seconds from segment start. None if unavailable."""
  log = next((seg_dir / n for n in ('rlog.zst', 'rlog.bz2', 'rlog', 'qlog.zst', 'qlog.bz2', 'qlog') if (seg_dir / n).exists()), None)
  if log is None:
    return None
  try:
    from openpilot.tools.lib.logreader import LogReader
  except ImportError:
    return None
  speed, yaw, roads = [], [], []
  t0 = None
  for msg in LogReader(str(log)):
    # not the log's first message: initData carries the time the route started, so only segment 0 would line up
    if t0 is None:
      if msg.which() != 'carState':
        continue
      t0 = msg.logMonoTime
    t = (msg.logMonoTime - t0) * 1e-9
    if msg.which() == 'carState':
      speed.append((t, msg.carState.vEgo))
    elif msg.which() == 'deviceMotion':
      yaw.append((t, -msg.deviceMotion.angularVelocityDevice.z))  # device z points down, as in signd
    elif msg.which() == 'liveMapDataSP':
      roads.append((t, msg.liveMapDataSP.roadName))
  if not speed:
    return None
  s = np.array(speed)
  y = np.array(yaw) if yaw else np.zeros((1, 2))
  return s[:, 0], s[:, 1], np.interp(s[:, 0], y[:, 0], y[:, 1]), roads


def load_labels(fn: Path) -> dict[tuple[str, int], list[tuple[int, int, int]]]:
  labels: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
  with open(fn) as f:
    for row in csv.DictReader(f):
      key = (row['route'], int(row['segment']))
      labels.setdefault(key, []).append((int(row['start_frame']), int(row['end_frame']), int(row['limit'])))
  return labels


def label_at(intervals: list[tuple[int, int, int]], frame: int) -> int | None:
  for start, end, limit in intervals:
    if start <= frame <= end:
      return limit
  return None


def score(timeline: list[dict]) -> dict:
  scored = [r for r in timeline if r['truth'] is not None]
  shown = [r for r in scored if r['speed_limit'] > 0]
  correct = [r for r in shown if r['speed_limit'] == r['truth']]
  posted = [r for r in scored if r['truth'] > 0]

  # seconds from the start of each labeled limit until the filter shows it
  delays = []
  current, start_t, found = None, 0., True
  for r in scored:
    if r['truth'] != current:
      if not found and current:
        delays.append(None)
      current, start_t, found = r['truth'], r['t'], False
    if not found and current and r['speed_limit'] == current:
      delays.append(r['t'] - start_t)
      found = True
  if not found and current:
    delays.append(None)
  hit = [d for d in delays if d is not None]

  return {
    'samples': len(scored),
    'precision': len(correct) / len(shown) if shown else None,  # when a limit is shown, how often it is right
    'coverage': sum(r['speed_limit'] == r['truth'] for r in posted) / len(posted) if posted else None,
    'limits_labeled': len(delays),
    'limits_found': len(hit),
    'median_time_to_limit': float(np.median(hit)) if hit else None,
    'wrong_values': sorted({(r['truth'], r['speed_limit']) for r in shown if r['speed_limit'] != r['truth']}),
  }


def annotate(rgb: np.ndarray, roi: tuple[int, int, int, int], dets: list[Detection], speed_limit: int, truth: int | None) -> np.ndarray:
  import cv2
  img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  h, w = img.shape[:2]
  cv2.rectangle(img, roi[:2], roi[2:], (128, 128, 128), 1)
  for d in dets:
    x, y, bw, bh = d.box
    p0, p1 = (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h))
    cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
    cv2.putText(img, f'{d.value} {d.confidence:.2f}', (p0[0], p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  text = f'limit {speed_limit or "-"}' + (f'  truth {truth}' if truth is not None else '')
  color = (255, 255, 255) if truth is None or truth == speed_limit else (0, 0, 255)
  (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
  cv2.rectangle(img, (10, 10), (30 + tw, 50 + th - 20), (0, 0, 0), -1)
  cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
  return img


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('route', type=Path)
  ap.add_argument('--camera', choices=list(CAMERA_FILES), default='road')
  ap.add_argument('--rate', type=float, default=2., help='Hz to sample the timeline at (and to run detection at with --fixed)')
  ap.add_argument('--new-track-time', type=float, help="override ReaderParams.new_track_time (0 = no fast reads of new signs)")
  ap.add_argument('--fixed', action='store_true', help='full detection at --rate instead of signd\'s scan + burst scheduling')
  ap.add_argument('--models', type=Path, help='directory with sign_detector.onnx and sign_classifier.onnx')
  ap.add_argument('--labels', type=Path, help='labels.csv to score against')
  ap.add_argument('--out', type=Path, default=Path('sign_eval_out'))
  ap.add_argument('--video', action='store_true', help='write an annotated mp4 per segment')
  ap.add_argument('--dump-rois', type=Path, help='save ROI crops as jpg (for building a dataset) instead of detecting')
  args = ap.parse_args()

  if args.video or args.dump_rois:
    try:
      import cv2
    except ImportError:
      print('--video and --dump-rois need opencv: uv pip install opencv-python-headless', file=sys.stderr)
      return 1
  from openpilot.sunnypilot.selfdrive.signd.detector import MODELS_DIR, SignDetector, nv12_to_rgb, roi_pixels
  from openpilot.sunnypilot.selfdrive.signd.reader import ReaderParams, SignReader

  segments = find_segments(args.route)
  if not segments:
    print(f'no segments found in {args.route}', file=sys.stderr)
    return 1

  detector = None
  if not args.dump_rois:
    models_dir = args.models or MODELS_DIR
    if not SignDetector.available(models_dir):
      print(f'no models in {models_dir}; use --models, or --dump-rois to collect training images', file=sys.stderr)
      return 1
    detector = SignDetector(args.camera, models_dir)

  labels = load_labels(args.labels) if args.labels else {}
  args.out.mkdir(parents=True, exist_ok=True)
  if args.dump_rois:
    args.dump_rois.mkdir(parents=True, exist_ok=True)

  sign_filter = SignFilter()
  filter_no_road = SignFilter()  # the hold rules without map road names, to compare
  reader_params = ReaderParams() if args.new_track_time is None else ReaderParams(new_track_time=args.new_track_time)
  reader = SignReader(detector, reader_params) if detector is not None and not args.fixed else None
  sample_every = max(1, round(CAMERA_FPS / args.rate))
  timeline: list[dict] = []
  for route, seg, seg_dir in segments:
    cam = next((seg_dir / n for n in CAMERA_FILES[args.camera] if (seg_dir / n).exists()), None)
    if cam is None:
      print(f'{seg_dir.name}: no {CAMERA_FILES[args.camera][0]}, skipping', file=sys.stderr)
      continue
    vehicle = load_vehicle_state(seg_dir)
    intervals = labels.get((route, seg), [])
    writer = None
    print(f'{seg_dir.name}: {cam.name}' + ('' if vehicle else ' (no vehicle state, distance/turn resets disabled)'))

    w, h = probe_size(cam)
    pending: list[tuple[int, Detection]] = []  # (frame, detection) since the last timeline sample
    for frame, nv12 in iter_frames(cam, CAMERA_FPS if reader is not None else args.rate):
      seg_t = frame / CAMERA_FPS
      x0, y0, x1, y1 = roi_pixels(args.camera, w, h)
      if args.dump_rois:
        rgb = nv12_to_rgb(nv12, w, w * h, (0, 0, w, h))
        cv2.imwrite(str(args.dump_rois / f'{route}--{seg}--{frame:04d}.jpg'), cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
        continue

      assert detector is not None
      v_ego: float | None = None  # unknown without logs: the filter's distance resets are off
      yaw_rate = 0.
      road = ''
      if vehicle is not None:
        v_ego = float(np.interp(seg_t, vehicle[0], vehicle[1]))
        yaw_rate = float(np.interp(seg_t, vehicle[0], vehicle[2]))
        road = road_at(vehicle[3], seg_t)
      t = seg * SEGMENT_SECONDS + seg_t
      if reader is not None:
        dets = reader.update(t, nv12, w, w * h, w, h) or []
      else:
        dets = detector.detect_nv12(nv12, w, w * h, w, h)
      pending += [(frame, d) for d in dets]
      state = sign_filter.update(t, dets, v_ego, yaw_rate, road)
      state_no_road = filter_no_road.update(t, dets, v_ego, yaw_rate)
      if frame % sample_every:
        continue
      found, pending = pending, []
      dets = [d for _, d in found]
      truth = label_at(intervals, frame)
      timeline.append({'route': route, 'segment': seg, 'frame': frame, 't': t,
                       'v_ego': v_ego or 0., 'road': road, 'detections': [asdict(d) | {'frame': f} for f, d in found],
                       'speed_limit': state.speed_limit, 'speed_limit_no_road': state_no_road.speed_limit, 'confidence': state.confidence, 'truth': truth})

      if args.video:
        img = annotate(nv12_to_rgb(nv12, w, w * h, (0, 0, w, h)), (x0, y0, x1, y1), dets, state.speed_limit, truth)
        if writer is None:
          fn = args.out / f'{route}--{seg}--{args.camera}.mp4'
          writer = cv2.VideoWriter(str(fn), cv2.VideoWriter.fourcc(*'mp4v'), args.rate, img.shape[1::-1])
        writer.write(img)

    if writer is not None:
      writer.release()

  if args.dump_rois:
    print(f'wrote ROI crops to {args.dump_rois}')
    return 0

  with open(args.out / 'timeline.json', 'w') as f:
    json.dump(timeline, f, indent=1)
  print(f'wrote {args.out / "timeline.json"} ({len(timeline)} samples)')

  if labels:
    result = score(timeline)
    if reader is not None:
      seconds = len(timeline) / args.rate
      result['runs_per_s'] = {k: round(v / seconds, 2) for k, v in reader.runs.items()}
    with open(args.out / 'score.json', 'w') as f:
      json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
  return 0


if __name__ == '__main__':
  sys.exit(main())
