"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Two-stage speed limit sign detector: a sign detector on a region of interest of the road frame,
then a value classifier on each detected sign crop.

Model contract (both ONNX, single input each, see README.md):
  detector:   input   float32 [1, 3, S, S], RGB in 0..1, letterboxed ROI (S read from the model, e.g. 640)
              output  either post-NMS boxes float32 [N, 6]: x1, y1, x2, y2 (input pixels), score, class
                      or raw YOLO-style predictions float32 [1, 4 + C, A]: cx, cy, w, h, then C class scores
                      (decoded and NMS'd here, since NMS ops don't run on tinygrad)
                      class 0 = speed limit sign, anything else is ignored
  classifier: input   float32 [1, 3, CLS_SIZE, CLS_SIZE] (or a dynamic batch dim), RGB in 0..1: the detected box plus a margin of 1/3 its width
              output  float32 [1, len(SPEED_LIMIT_VALUES) + 1] logits, last index = reject
                      (advisory, end of zone, school zone, minimum, truck, unreadable...)
"""
import os
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.selfdrive.signd import SPEED_LIMIT_VALUES
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection

MODELS_DIR = Path(os.getenv('SIGND_MODELS_DIR', Path(__file__).parent / 'models'))
DET_SIZE = 640  # default, the detector's input size is read from the model
CLS_SIZE = 64
SIGN_CLASS = 0
NMS_IOU = 0.5

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


def softmax(x: np.ndarray) -> np.ndarray:
  e = np.exp(x - x.max(axis=-1, keepdims=True))
  return e / e.sum(axis=-1, keepdims=True)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
  """Bilinear resize in numpy (opencv isn't available on device)"""
  ih, iw = img.shape[:2]
  ys = np.clip((np.arange(h) + 0.5) * ih / h - 0.5, 0, ih - 1)
  xs = np.clip((np.arange(w) + 0.5) * iw / w - 0.5, 0, iw - 1)
  y0, x0 = ys.astype(int), xs.astype(int)
  y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
  fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
  img = img.astype(np.float32)
  top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
  bottom = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
  return (top * (1 - fy) + bottom * fy).astype(np.uint8)


def nv12_to_rgb(data: np.ndarray, stride: int, uv_offset: int, roi: tuple[int, int, int, int]) -> np.ndarray:
  """Converts the roi x0, y0, x1, y1 (even aligned, see roi_pixels) of an NV12 VisionIPC buffer to RGB, BT.601 full range"""
  x0, y0, x1, y1 = roi
  y = data[:uv_offset].reshape(-1, stride)[y0:y1, x0:x1].astype(np.float32)
  uv = data[uv_offset:].reshape(-1, stride)[y0 // 2:y1 // 2, x0:x1].reshape(-1, (x1 - x0) // 2, 2).astype(np.float32) - 128.
  uv = uv.repeat(2, axis=0).repeat(2, axis=1)
  u, v = uv[..., 0], uv[..., 1]
  rgb = np.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
  """Resize keeping aspect ratio and pad to size x size. Returns image, scale and padding."""
  h, w = img.shape[:2]
  scale = size / max(h, w)
  nh, nw = round(h * scale), round(w * scale)
  out = np.full((size, size, 3), 114, dtype=np.uint8)
  pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
  out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resize(img, nw, nh)
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
    return self._run(x)


class SignDetector:
  def __init__(self, camera: str = 'road', models_dir: Path = MODELS_DIR, min_score: float = 0.3):
    self.camera = camera
    self.min_score = min_score
    self.detector = OnnxModel(models_dir / 'sign_detector.onnx')
    self.classifier = OnnxModel(models_dir / 'sign_classifier.onnx')
    size = self.detector.input_shape[-1]
    self.det_size = size if isinstance(size, int) else DET_SIZE

  @staticmethod
  def available(models_dir: Path = MODELS_DIR) -> bool:
    return (models_dir / 'sign_detector.onnx').exists() and (models_dir / 'sign_classifier.onnx').exists()

  def detect(self, rgb: np.ndarray) -> list[Detection]:
    """rgb: full camera frame, HxWx3 uint8"""
    h, w = rgb.shape[:2]
    rx0, ry0, rx1, ry1 = roi_pixels(self.camera, w, h)
    return self.detect_roi(rgb[ry0:ry1, rx0:rx1], w, h)

  def detect_roi(self, roi: np.ndarray, w: int, h: int) -> list[Detection]:
    """roi: the camera's ROI (see roi_pixels) as HxWx3 uint8 RGB; w, h: full frame size. Boxes are normalized to the full frame."""
    rx0, ry0, _, _ = roi_pixels(self.camera, w, h)
    rh, rw = roi.shape[:2]

    img, scale, pad_x, pad_y = letterbox(roi, self.det_size)
    boxes = decode_boxes(self.detector(to_nchw(img)), self.min_score)
    boxes = boxes[boxes[:, 5] == SIGN_CLASS]
    if len(boxes) == 0:
      return []

    # back to ROI pixels
    xyxy = (boxes[:, :4] - [pad_x, pad_y, pad_x, pad_y]) / scale
    xyxy = np.clip(xyxy, 0, [rw, rh, rw, rh]).astype(int)
    keep = (xyxy[:, 2] - xyxy[:, 0] >= 8) & (xyxy[:, 3] - xyxy[:, 1] >= 8)
    xyxy, scores = xyxy[keep], boxes[keep, 4]
    if len(xyxy) == 0:
      return []

    crops = []
    for x0, y0, x1, y1 in xyxy:
      m = max(6, (x1 - x0) // 3)  # same context margin the classifier was trained with
      crops.append(to_nchw(resize(roi[max(0, y0 - m):min(rh, y1 + m), max(0, x0 - m):min(rw, x1 + m)], CLS_SIZE, CLS_SIZE)))
    # one crop per call: tinygrad's OnnxRunner fixes dynamic dims on first use
    probs = softmax(np.concatenate([self.classifier(c) for c in crops]))

    detections = []
    for (x0, y0, x1, y1), score, p in zip(xyxy, scores, probs, strict=True):
      idx = int(p.argmax())
      if idx >= len(SPEED_LIMIT_VALUES):  # reject class
        continue
      box = (float((rx0 + x0) / w), float((ry0 + y0) / h), float((x1 - x0) / w), float((y1 - y0) / h))
      detections.append(Detection(SPEED_LIMIT_VALUES[idx], float(score * p[idx]), box))
    return detections
