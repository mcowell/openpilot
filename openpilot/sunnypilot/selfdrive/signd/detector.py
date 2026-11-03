"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Two-stage speed limit sign detector: a sign detector on a region of interest of the road frame,
then a value classifier on each detected sign crop.

Model contract (both ONNX, single input each, see README.md):
  detector:   input   float32 [1, 3, H, W], RGB in 0..1: the ROI at half resolution, padded (H, W read from the model; sized
                      to the half-res ROI, e.g. 352x544 for the 3X road camera, it needs no resize)
              output  either post-NMS boxes float32 [N, 6]: x1, y1, x2, y2 (input pixels), score, class
                      or raw YOLO-style predictions float32 [1, 4 + C, A]: cx, cy, w, h, then C class scores
                      (decoded and NMS'd here, since NMS ops don't run on tinygrad)
                      class 0 = speed limit sign, anything else is ignored
  classifier: input   float32 [1, 3, CLS_H, CLS_W], RGB in 0..1: see classifier_crop(), the sign panel plus the
                      area below it where tabs (AREA, BEGINS, ENDS) hang
              output  float32 [1, len(SPEED_LIMIT_VALUES) + 2] logits: the values, then reject
                      (advisory, school zone, minimum, truck, unreadable...), then end of zone ("ENDS / FIN" tab)
"""
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.selfdrive.signd import SPEED_LIMIT_VALUES
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection

MODELS_DIR = Path(os.getenv('SIGND_MODELS_DIR', Path(__file__).parent / 'models'))
DET_SIZE = 640  # default, the detector's input size is read from the model
CLS_W, CLS_H = 64, 96
# classifier crop around the detected panel box, in panel widths/heights: room for the tabs below
CLS_MARGIN_X, CLS_MARGIN_TOP, CLS_MARGIN_BOTTOM = 0.5, 0.2, 1.2
REJECT, END = len(SPEED_LIMIT_VALUES), len(SPEED_LIMIT_VALUES) + 1
SIGN_CLASS = 0
NMS_IOU = 0.5
LOCAL_SIGN_PX = 30  # target panel height in the local detector's input, px
TILE_SCALE = 0.4  # tiles see the ROI at the scale of the full scan (416 px over the 3X road ROI): what it was trained on
TILE_OVERLAP = 130  # ROI px; more than the largest panel, so every sign is whole in some tile
ENTRY_Y, ENTRY_X1 = 0.6, 0.21  # entry tiles: vertical centre, and left edge of the second one, as ROI fractions

# Normalized (x0, y0, x1, y1) region where roadside signs appear, per camera.
# Right side of the road and above the horizon; signs on the left of divided highways are out of scope for v1.
# The road camera is the default: on the 3X the wide (fisheye) camera's right edge is blocked by the mount
# and signs there are small and distorted.
ROIS = {
  'road': (0.45, 0.0, 1.0, 0.55),
  'wide': (0.45, 0.15, 0.75, 0.55),
}


def roi_pixels(camera: str, w: int, h: int) -> tuple[int, int, int, int]:
  x0, y0, x1, y1 = ROIS[camera]
  # even aligned so the ROI maps onto whole NV12 chroma samples
  return int(x0 * w) & ~1, int(y0 * h) & ~1, int(x1 * w) & ~1, int(y1 * h) & ~1


def classifier_crop(img: np.ndarray, box) -> np.ndarray:
  """CLS_H x CLS_W crop for the classifier from a panel box (x0, y0, x1, y1 in img pixels); parts outside img are grey"""
  x0, y0, x1, y1 = box
  bw, bh = x1 - x0, y1 - y0
  cx0, cx1 = int(round(x0 - CLS_MARGIN_X * bw)), int(round(x1 + CLS_MARGIN_X * bw))
  cy0, cy1 = int(round(y0 - CLS_MARGIN_TOP * bh)), int(round(y1 + CLS_MARGIN_BOTTOM * bh))
  ih, iw = img.shape[:2]
  out = np.full((cy1 - cy0, cx1 - cx0, 3), 114, dtype=np.uint8)
  sx0, sy0, sx1, sy1 = max(0, cx0), max(0, cy0), min(iw, cx1), min(ih, cy1)
  out[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0] = img[sy0:sy1, sx0:sx1]
  return resize(out, CLS_W, CLS_H)


def softmax(x: np.ndarray) -> np.ndarray:
  e = np.exp(x - x.max(axis=-1, keepdims=True))
  return e / e.sum(axis=-1, keepdims=True)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
  """Bilinear resize in numpy (opencv isn't available on device), separable: rows first, then columns"""
  ih, iw = img.shape[:2]
  ys = np.clip((np.arange(h) + 0.5) * ih / h - 0.5, 0, ih - 1)
  xs = np.clip((np.arange(w) + 0.5) * iw / w - 0.5, 0, iw - 1)
  y0, x0 = ys.astype(np.intp), xs.astype(np.intp)
  y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
  fy, fx = (ys - y0).astype(np.float32)[:, None, None], (xs - x0).astype(np.float32)[None, :, None]
  top, bottom = img[y0].astype(np.float32), img[y1].astype(np.float32)
  rows = top + (bottom - top) * fy
  left, right = rows[:, x0], rows[:, x1]
  return (left + (right - left) * fx).astype(np.uint8)


def nv12_to_rgb(data: np.ndarray, stride: int, uv_offset: int, roi: tuple[int, int, int, int]) -> np.ndarray:
  """Converts the roi x0, y0, x1, y1 (even aligned, see roi_pixels) of an NV12 VisionIPC buffer to RGB, BT.601 full range"""
  x0, y0, x1, y1 = roi
  y = data[:uv_offset].reshape(-1, stride)[y0:y1, x0:x1].astype(np.float32)
  uv = data[uv_offset:].reshape(-1, stride)[y0 // 2:y1 // 2, x0:x1].reshape(-1, (x1 - x0) // 2, 2).astype(np.float32) - 128.
  uv = uv.repeat(2, axis=0).repeat(2, axis=1)
  u, v = uv[..., 0], uv[..., 1]
  rgb = np.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def nv12_to_rgb_half(data: np.ndarray, stride: int, uv_offset: int, roi: tuple[int, int, int, int]) -> np.ndarray:
  """Half-resolution RGB of the roi: luma averaged 2x2 onto the chroma grid, so only a quarter of the pixels are converted.
  Doubles as an area downscale, which keeps small signs better than skipping pixels."""
  x0, y0, x1, y1 = roi
  h2, w2 = (y1 - y0) // 2, (x1 - x0) // 2
  yp = data[:uv_offset].reshape(-1, stride)[y0:y1, x0:x1]
  y = (yp[0::2, 0::2].astype(np.uint16) + yp[0::2, 1::2] + yp[1::2, 0::2] + yp[1::2, 1::2]) * np.float32(0.25)
  uv = data[uv_offset:].reshape(-1, stride)[y0 // 2:y1 // 2, x0:x1].reshape(h2, w2, 2).astype(np.float32) - 128.
  u, v = uv[..., 0], uv[..., 1]
  rgb = np.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def rgb_to_nv12(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
  """Inverse of nv12_to_rgb (BT.601 full range), for running RGB images through the device code path.
  Returns data, stride, uv_offset."""
  f = rgb.astype(np.float32)
  h, w = f.shape[:2]
  y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
  sub = f.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
  ys = y.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
  uv = np.stack([(sub[..., 2] - ys) / 1.772 + 128., (sub[..., 0] - ys) / 1.402 + 128.], axis=-1).reshape(h // 2, w)
  data = np.concatenate([np.clip(np.round(y), 0, 255).astype(np.uint8).ravel(), np.clip(np.round(uv), 0, 255).astype(np.uint8).ravel()])
  return data, w, w * h


def letterbox(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
  """Fit img into w x h keeping aspect ratio (never upscaling), padded with grey. Returns image, scale and padding.
  When the model input is sized to the image, this is only a pad: no resize on the CPU."""
  ih, iw = img.shape[:2]
  scale = min(1., w / iw, h / ih)
  nh, nw = round(ih * scale), round(iw * scale)
  out = np.full((h, w, 3), 114, dtype=np.uint8)
  pad_y, pad_x = (h - nh) // 2, (w - nw) // 2
  out[pad_y:pad_y + nh, pad_x:pad_x + nw] = img if scale == 1. else resize(img, nw, nh)
  return out, scale, pad_x, pad_y


def to_nchw(img: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.


def nms(xyxy: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
  order = scores.argsort()[::-1]
  keep = []
  areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
  while len(order):
    i = order[0]
    keep.append(int(i))
    xx1, yy1 = np.maximum(xyxy[i, 0], xyxy[order[1:], 0]), np.maximum(xyxy[i, 1], xyxy[order[1:], 1])
    xx2, yy2 = np.minimum(xyxy[i, 2], xyxy[order[1:], 2]), np.minimum(xyxy[i, 3], xyxy[order[1:], 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) < iou]
  return keep


def decode_boxes(out: np.ndarray, min_score: float) -> np.ndarray:
  """Detector output (either contract form) -> [N, 6] x1, y1, x2, y2, score, class after NMS"""
  if out.ndim == 3 and out.shape[2] != 6:  # raw YOLO [1, 4 + C, A]; post-NMS exports are [1, N, 6]
    pred = out[0].T
    cls = pred[:, 4:].argmax(1)
    score = pred[:, 4:].max(1)
    pred, cls, score = pred[score >= min_score], cls[score >= min_score], score[score >= min_score]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(xyxy, score)
    return np.concatenate([xyxy[keep], score[keep, None], cls[keep, None].astype(np.float32)], axis=1)
  boxes = out.reshape(-1, 6)
  return boxes[boxes[:, 4] >= min_score]


class OnnxModel:
  """Runs a single-input ONNX model with onnxruntime when installed (PC/Mac), else with tinygrad (comma device)."""
  def __init__(self, path: Path):
    self.path = path
    try:
      import onnxruntime as ort
      sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
      inp = sess.get_inputs()[0]
      self.input_name, self.input_shape = inp.name, inp.shape
      self._run = lambda x: sess.run(None, {self.input_name: x})[0]
    except ImportError:
      from tinygrad import Tensor
      from tinygrad.nn.onnx import OnnxRunner
      runner = OnnxRunner(str(path))
      self.input_name = next(iter(runner.graph_inputs))
      self.input_shape = list(runner.graph_inputs[self.input_name].shape)
      out_name = runner.graph_outputs[0]
      self._run = lambda x: runner({self.input_name: Tensor(x)})[out_name].numpy()

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Returns the first output"""
    return np.asarray(self._run(x))


class CompiledModel:
  """Runs a model compiled by SConscript (tinygrad_repo/examples/openpilot/compile_onnx.py), as modeld does on device."""
  def __init__(self, path: Path):
    from tinygrad.tensor import Tensor
    from openpilot.selfdrive.modeld.helpers import load_oob

    self.path = path
    jits = load_oob(path)
    self.model_run = jits['run']
    (self.input_name, (shape, dtype, device)), = jits['input_specs'].items()
    self.input_shape = list(shape)
    self.input_np = np.zeros(shape, dtype=dtype)
    self.input_host = Tensor(self.input_np, device='NPY')._buffer()
    self.inputs = {self.input_name: Tensor(self.input_np, device=device).realize()}
    self.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                    for name, (shape, dtype, device) in jits['output_specs'].items()}
    self.output_name = next(iter(self.outputs))

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Returns the first output"""
    self.input_np[...] = x
    self.inputs[self.input_name]._buffer().copy_from(self.input_host)
    self.model_run(output_buffers=self.outputs, **self.inputs)
    return self.outputs[self.output_name].numpy().astype(np.float32)


def load_model(models_dir: Path, name: str) -> 'CompiledModel | OnnxModel':
  """The compiled model when the build made one, else the ONNX model (PC/Mac tools and tests)"""
  pkl = models_dir / f'{name}_tinygrad.pkl'
  return CompiledModel(pkl) if pkl.exists() else OnnxModel(models_dir / f'{name}.onnx')


@dataclass(frozen=True)
class Candidate:
  """A detected sign, in full-resolution ROI pixels. value: the speed, 0 for end of zone, None if the classifier rejected it
  (often just too far away to read yet)."""
  box: tuple[int, int, int, int]  # x0, y0, x1, y1
  score: float  # detector score
  value: int | None
  confidence: float  # sqrt(detector score) * classifier probability, 0 when rejected


@dataclass(frozen=True)
class Window:
  """A local-detector input prepared on the CPU, and how its boxes map back to ROI pixels"""
  x: np.ndarray  # NCHW input
  scale: float  # input pixels per ROI pixel
  pad_x: int
  pad_y: int
  x0: int  # window origin, ROI pixels
  y0: int


def best_overlap(found: np.ndarray, box) -> np.ndarray | None:
  """The highest-scoring of found ([N, 5]: x0, y0, x1, y1, score) that overlaps box, or None"""
  if len(found) == 0:
    return None
  x0, y0, x1, y1 = box
  ix = np.clip(np.minimum(found[:, 2], x1) - np.maximum(found[:, 0], x0), 0, None)
  iy = np.clip(np.minimum(found[:, 3], y1) - np.maximum(found[:, 1], y0), 0, None)
  overlap = ix * iy
  if overlap.max() <= 0.:
    return None
  return found[int(np.argmax(np.where(overlap > 0., found[:, 4], -1.)))]


class SignDetector:
  def __init__(self, camera: str = 'road', models_dir: Path = MODELS_DIR, min_score: float = 0.3):
    self.camera = camera
    self.min_score = min_score
    self.detector = load_model(models_dir, 'sign_detector')
    self.classifier = load_model(models_dir, 'sign_classifier')
    dh, dw = self.detector.input_shape[-2:]
    self.det_h, self.det_w = (dh, dw) if isinstance(dh, int) and isinstance(dw, int) else (DET_SIZE, DET_SIZE)
    # optional small detector for re-finding a tracked sign in a window around it (see refine())
    has_local = any((models_dir / f'sign_detector_local{ext}').exists() for ext in ('_tinygrad.pkl', '.onnx'))
    self.local = load_model(models_dir, 'sign_detector_local') if has_local else None
    self.local_size = int(self.local.input_shape[-1]) if self.local is not None else 0

  @staticmethod
  def available(models_dir: Path = MODELS_DIR, compiled: bool | None = None) -> bool:
    """compiled: require the build's compiled models (default on device, where uncompiled ONNX is far too slow)"""
    if compiled is None:
      compiled = os.path.isfile('/AGNOS')
    ext = '_tinygrad.pkl' if compiled else '.onnx'
    return all((models_dir / f'{name}{ext}').exists() for name in ('sign_detector', 'sign_classifier'))

  def detect(self, rgb: np.ndarray) -> list[Detection]:
    """rgb: full camera frame, HxWx3 uint8. Converted to NV12 so it takes the same path as camera frames."""
    data, stride, uv_offset = rgb_to_nv12(rgb)
    return self.detect_nv12(data, stride, uv_offset, rgb.shape[1], rgb.shape[0])

  def detect_nv12(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int) -> list[Detection]:
    """data: NV12 frame (VisionIPC buffer layout), w x h. Boxes are normalized to the full frame."""
    return [d for c in self.scan(data, stride, uv_offset, w, h) if (d := self.to_detection(c, w, h)) is not None]

  def to_detection(self, c: Candidate, w: int, h: int) -> Detection | None:
    if c.value is None:
      return None
    rx0, ry0, _, _ = roi_pixels(self.camera, w, h)
    x0, y0, x1, y1 = c.box
    box = (float((rx0 + x0) / w), float((ry0 + y0) / h), float((x1 - x0) / w), float((y1 - y0) / h))
    # an end-of-zone sign is reported as value 0: once confirmed, the filter drops the held limit
    return Detection(c.value, c.confidence, box)

  def scan(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int) -> list[Candidate]:
    """Full detector over the ROI (half resolution, letterboxed), then the classifier on each box"""
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    rw, rh = rx1 - rx0, ry1 - ry0
    img, scale, pad_x, pad_y = letterbox(nv12_to_rgb_half(data, stride, uv_offset, (rx0, ry0, rx1, ry1)), self.det_w, self.det_h)
    boxes = decode_boxes(self.detector(to_nchw(img)), self.min_score)
    boxes = boxes[boxes[:, 5] == SIGN_CLASS]
    # back to full-resolution ROI pixels
    xyxy = np.clip((boxes[:, :4] - [pad_x, pad_y, pad_x, pad_y]) / scale * 2, 0, [rw, rh, rw, rh]).astype(int)
    return self._classify(data, stride, uv_offset, w, h, xyxy, boxes[:, 4])

  # Small GPU jobs, split into a CPU prepare (done while the GPU is busy with the driving model) and a GPU run (done in
  # the gap after the driving model finishes a frame). Each run is ~10-15 ms on a 3X, so it fits in that gap.

  def tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Overlapping square windows covering the ROI (ROI pixels) for the local detector at TILE_SCALE: together, a scan"""
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    rw, rh = rx1 - rx0, ry1 - ry0
    side = min(rw, rh, int(self.local_size / TILE_SCALE)) & ~3
    def starts(n):
      count = max(1, math.ceil((n - side) / (side - TILE_OVERLAP)) + 1)
      return sorted({int(round(i * (n - side) / max(1, count - 1))) & ~3 for i in range(count)})
    return [(x, y, x + side, y + side) for y in starts(rh) for x in starts(rw)]

  def entry_tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """The tiles where signs first appear: small and far, left of the ROI near the vanishing point (on held-out drives
    first sightings were at x 0-450, y 180-550 of the 1062x664 3X road ROI). Swept more often than the rest."""
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    rw, rh = rx1 - rx0, ry1 - ry0
    side = min(rw, rh, int(self.local_size / TILE_SCALE)) & ~3
    y = int(np.clip(ENTRY_Y * rh - side / 2, 0, rh - side)) & ~3
    return [(x, y, x + side, y + side) for x in sorted({0, int(np.clip(ENTRY_X1 * rw, 0, rw - side)) & ~3})]

  def prepare_tile(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, rect) -> Window:
    rx0, ry0, _, _ = roi_pixels(self.camera, w, h)
    x0, y0, x1, y1 = rect
    half = nv12_to_rgb_half(data, stride, uv_offset, (rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1))
    img, scale, pad_x, pad_y = letterbox(half, self.local_size, self.local_size)
    return Window(to_nchw(img), scale / 2, pad_x, pad_y, x0, y0)

  def prepare_local(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> Window:
    """Window around box (ROI pixels, e.g. a track's predicted box), sized so the sign is ~LOCAL_SIGN_PX in the input"""
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    rw, rh = rx1 - rx0, ry1 - ry0
    x0, y0, x1, y1 = box
    side = int(np.clip((y1 - y0) * self.local_size / LOCAL_SIGN_PX, self.local_size, min(rw, rh)))
    cx, cy = (x0 + x1) / 2, (y0 + y1 * 2) / 3  # a little low: the window includes the tabs below the panel
    wx0 = int(np.clip(cx - side / 2, 0, rw - side)) & ~1
    wy0 = int(np.clip(cy - side / 2, 0, rh - side)) & ~1
    side &= ~1
    region = nv12_to_rgb(data, stride, uv_offset, (rx0 + wx0, ry0 + wy0, rx0 + wx0 + side, ry0 + wy0 + side))
    img, scale, pad_x, pad_y = letterbox(region, self.local_size, self.local_size)
    return Window(to_nchw(img), scale, pad_x, pad_y, wx0, wy0)

  def run_window(self, win: Window, w: int, h: int) -> np.ndarray:
    """Local detector on a prepared window: [N, 5] boxes x0, y0, x1, y1 (ROI pixels) and score"""
    assert self.local is not None
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    boxes = decode_boxes(self.local(win.x), self.min_score)
    boxes = boxes[boxes[:, 5] == SIGN_CLASS]
    xyxy = (boxes[:, :4] - [win.pad_x, win.pad_y, win.pad_x, win.pad_y]) / win.scale + [win.x0, win.y0, win.x0, win.y0]
    xyxy = np.clip(xyxy, 0, [rx1 - rx0, ry1 - ry0] * 2)
    return np.concatenate([xyxy, boxes[:, 4:5]], axis=1)

  def prepare_crop(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> np.ndarray | None:
    """Classifier input for box (ROI pixels), at full resolution, converting only the crop's region (clipped to the ROI,
    as in training). None if the box is too small to classify."""
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    rw, rh = rx1 - rx0, ry1 - ry0
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
      return None
    cx0, cy0 = max(0, int(x0 - CLS_MARGIN_X * bw)) & ~1, max(0, int(y0 - CLS_MARGIN_TOP * bh)) & ~1
    cx1, cy1 = min(rw, int(x1 + CLS_MARGIN_X * bw) + 2) & ~1, min(rh, int(y1 + CLS_MARGIN_BOTTOM * bh) + 2) & ~1
    region = nv12_to_rgb(data, stride, uv_offset, (rx0 + cx0, ry0 + cy0, rx0 + cx1, ry0 + cy1))
    return to_nchw(classifier_crop(region, (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)))

  def run_classifier(self, x: np.ndarray) -> tuple[int | None, float]:
    """value (speed, 0 for end of zone, None for reject) and its probability"""
    p = softmax(self.classifier(x))[0]
    idx = int(p.argmax())
    return (None if idx == REJECT else (SPEED_LIMIT_VALUES[idx] if idx < REJECT else 0)), float(p[idx])

  def candidate(self, box, score: float, value: int | None, prob: float) -> Candidate:
    """confidence = sqrt(detector score) * classifier probability: the classifier (which has a reject class) says whether
    the sign can be read; the detector score is mostly "is this a sign", already gated by min_score, and the small local
    detector scores lower than the full scan for the same sign, so it is weighted less."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return Candidate((x0, y0, x1, y1), float(score), value, float(math.sqrt(score) * prob) if value is not None else 0.)

  def refine(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> Candidate | None:
    """Re-find a tracked sign near box (ROI pixels, e.g. predicted from its motion) with the local detector, then classify
    it. None if the sign isn't found there."""
    found = best_overlap(self.run_window(self.prepare_local(data, stride, uv_offset, w, h, box), w, h), box)
    if found is None:
      return None
    x = self.prepare_crop(data, stride, uv_offset, w, h, found[:4])
    return None if x is None else self.candidate(found[:4], found[4], *self.run_classifier(x))

  def _classify(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, xyxy: np.ndarray, scores: np.ndarray) -> list[Candidate]:
    out = []
    for box, score in zip(xyxy, scores, strict=True):
      x = self.prepare_crop(data, stride, uv_offset, w, h, box)
      if x is not None:
        out.append(self.candidate(box, score, *self.run_classifier(x)))  # one crop per call: tinygrad fixes dims
    return out
