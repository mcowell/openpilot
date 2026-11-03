"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from pathlib import Path

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.signd.detector import (CLS_H, CLS_W, DET_SIZE, SignDetector, classifier_crop, decode_boxes, letterbox,
                                                           nv12_to_rgb, nv12_to_rgb_half, resize, roi_pixels)
from openpilot.sunnypilot.selfdrive.signd.tools.sign_eval import score

# constant-output models from tests/fake_models.py: one sign reading 60, at the same spot every frame
FAKE_MODELS = Path(__file__).parent / 'data' / 'fake_models'
FAKE_MODELS_END = Path(__file__).parent / 'data' / 'fake_models_end'  # classifier says end of zone


def rgb_to_nv12(rgb: np.ndarray, stride: int) -> tuple[np.ndarray, int]:
  """Inverse of nv12_to_rgb (BT.601 full range), padded to stride"""
  rgb = rgb.astype(np.float32)
  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = (b - y) / 1.772 + 128.
  v = (r - y) / 1.402 + 128.
  h, w = y.shape
  y_plane = np.zeros((h, stride), np.uint8)
  y_plane[:, :w] = np.clip(np.round(y), 0, 255)
  uv_plane = np.zeros((h // 2, stride), np.uint8)
  uv_plane[:, 0:w:2] = np.clip(np.round(u[::2, ::2]), 0, 255)
  uv_plane[:, 1:w:2] = np.clip(np.round(v[::2, ::2]), 0, 255)
  return np.concatenate([y_plane.ravel(), uv_plane.ravel()]), y_plane.size


class TestSignDetector(OpenpilotTestCase):
  def test_fake_models_end_to_end(self):
    det = SignDetector('road', FAKE_MODELS)

    w, h = 1928, 1208
    dets = det.detect(np.zeros((h, w, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].value == 60
    assert 0.8 < dets[0].confidence <= 0.9

    # box maps back inside the ROI in full-frame coordinates
    rx0, ry0, rx1, ry1 = roi_pixels('road', w, h)
    x, y, bw, bh = dets[0].box
    assert rx0 <= x * w and (x + bw) * w <= rx1
    assert ry0 <= y * h and (y + bh) * h <= ry1

  def test_refine_finds_sign_near_prediction(self):
    # the fake local detector always reports a box at (86, 60)-(106, 85) of its 192 px input
    det = SignDetector('road', FAKE_MODELS)
    w, h = 1928, 1208
    data, uv_offset = rgb_to_nv12(np.zeros((h, w, 3), dtype=np.uint8), w)
    c = det.refine(data, w, uv_offset, w, h, (500, 100, 520, 125))
    assert c is not None and c.value == 60
    # a 25 px sign gets the minimum 192 px window (no resize), centred a little low, so the box maps back 1:1
    assert c.box == (500, 80, 520, 105)

  def test_refine_ignores_box_away_from_prediction(self):
    det = SignDetector('road', FAKE_MODELS)
    w, h = 1928, 1208
    data, uv_offset = rgb_to_nv12(np.zeros((h, w, 3), dtype=np.uint8), w)
    # predicted box in the window's corner: the local detector's box doesn't overlap it
    assert det.refine(data, w, uv_offset, w, h, (420, 30, 430, 42)) is None

  def test_available_requires_compiled_models_on_device(self):
    # the fake models are ONNX only: fine for PC tools and tests, not for the device
    assert SignDetector.available(FAKE_MODELS, compiled=False)
    assert not SignDetector.available(FAKE_MODELS, compiled=True)

  def test_onnx_fallback_without_compiled_models(self):
    det = SignDetector('road', FAKE_MODELS)
    assert type(det.detector).__name__ == 'OnnxModel' and type(det.classifier).__name__ == 'OnnxModel'

  def test_end_of_zone_reported_as_zero(self):
    dets = SignDetector('road', FAKE_MODELS_END).detect(np.zeros((1208, 1928, 3), dtype=np.uint8))
    assert [x.value for x in dets] == [0]

  def test_classifier_crop_includes_tabs(self):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[100:200, 100:180] = 255  # 80x100 panel
    img[210:300, 100:180] = 50  # tab plates below it
    crop = classifier_crop(img, (100, 100, 180, 200))
    assert crop.shape == (CLS_H, CLS_W, 3)
    # panel spans the middle half horizontally; tabs fill the lower part of the crop
    assert crop[CLS_H // 4, CLS_W // 2].min() > 200
    assert abs(int(crop[int(CLS_H * 0.75), CLS_W // 2, 0]) - 50) < 10
    # off the image edge is padded grey, not stretched
    edge = classifier_crop(img, (0, 0, 80, 100))
    assert edge.shape == (CLS_H, CLS_W, 3) and edge[0, 0, 0] == 114

  def test_detect_nv12_matches_detect(self):
    # VisionIPC buffers have rows padded to a stride wider than the frame
    det = SignDetector('road', FAKE_MODELS)
    w, h = 1928, 1208
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    data, uv_offset = rgb_to_nv12(frame, 2048)
    assert det.detect(frame) == det.detect_nv12(data, 2048, uv_offset, w, h)

  def test_nv12_to_rgb_half(self):
    rgb = np.random.default_rng(1).integers(0, 255, (48, 64, 3)).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb[::2, ::2], 2, axis=0), 2, axis=1)  # constant 2x2 blocks: no chroma subsampling loss
    data, uv_offset = rgb_to_nv12(rgb, 80)
    out = nv12_to_rgb_half(data, 80, uv_offset, (8, 4, 40, 28))
    expected = rgb[4:28:2, 8:40:2].astype(int)
    assert out.shape == expected.shape
    assert np.abs(out.astype(int) - expected).max() <= 3

  def test_roi_pixels_even(self):
    for camera in ('wide', 'road'):
      for w, h in ((1928, 1208), (1344, 760), (1163, 875)):
        assert all(v % 2 == 0 for v in roi_pixels(camera, w, h))

  def test_nv12_to_rgb(self):
    rng = np.random.default_rng(0)
    w, h, stride = 64, 32, 128
    # 2x2 blocks of one color, so chroma subsampling is lossless
    rgb = rng.integers(0, 256, (h // 2, w // 2, 3), dtype=np.uint8).repeat(2, axis=0).repeat(2, axis=1)
    data, uv_offset = rgb_to_nv12(rgb, stride)
    roi = (8, 4, 40, 28)
    out = nv12_to_rgb(data, stride, uv_offset, roi)
    expected = rgb[4:28, 8:40].astype(int)
    assert out.shape == expected.shape
    assert np.abs(out.astype(int) - expected).max() <= 3

  def test_resize(self):
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert resize(img, 6, 4).tolist() == img.tolist()
    assert resize(np.full((10, 20, 3), 77, np.uint8), 7, 3).tolist() == np.full((3, 7, 3), 77, np.uint8).tolist()

  def test_decode_raw_yolo(self):
    # [1, 4 + C, A]: two overlapping boxes on one sign, one other sign, one low-score anchor
    pred = np.array([
      # cx,  cy,  w,  h, cls0, cls1
      [100, 100, 20, 40, 0.9, 0.0],
      [102, 101, 20, 40, 0.8, 0.0],
      [300, 120, 30, 50, 0.7, 0.1],
      [500, 500, 10, 10, 0.1, 0.0],
    ], dtype=np.float32).T[None]
    boxes = decode_boxes(pred, 0.3)
    assert boxes.shape == (2, 6)
    np.testing.assert_allclose(boxes[0], [90, 80, 110, 120, 0.9, 0], atol=1e-5)
    np.testing.assert_allclose(boxes[1], [285, 95, 315, 145, 0.7, 0], atol=1e-5)

  def test_decode_post_nms(self):
    boxes = decode_boxes(np.array([[1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.1, 0]], np.float32), 0.3)
    assert boxes.shape == (1, 6)

  def test_letterbox_keeps_aspect(self):
    img, scale, pad_x, pad_y = letterbox(np.zeros((600, 1200, 3), dtype=np.uint8), DET_SIZE, DET_SIZE)
    assert img.shape == (DET_SIZE, DET_SIZE, 3)
    assert scale == DET_SIZE / 1200
    assert pad_x == 0 and pad_y == (DET_SIZE - 320) // 2

  def test_letterbox_pads_without_upscaling(self):
    src = np.full((332, 531, 3), 7, dtype=np.uint8)
    img, scale, pad_x, pad_y = letterbox(src, 544, 352)
    assert img.shape == (352, 544, 3) and scale == 1.
    assert (pad_x, pad_y) == (6, 10)
    assert (img[pad_y:pad_y + 332, pad_x:pad_x + 531] == 7).all() and img[0, 0, 0] == 114


class TestScore(OpenpilotTestCase):
  def test_score(self):
    rows = [{'t': t, 'truth': truth, 'speed_limit': shown} for t, truth, shown in [
      (0., 50, 0), (1., 50, 50), (2., 50, 50),   # found after 1 s
      (3., 80, 50), (4., 80, 70), (5., 80, 80),  # stale, then a misread, then found after 2 s
      (6., None, 100),                           # unlabeled, not scored
    ]]
    s = score(rows)
    assert s['samples'] == 6
    assert s['precision'] == 3 / 5
    assert s['coverage'] == 3 / 6
    assert s['limits_labeled'] == 2 and s['limits_found'] == 2
    assert s['median_time_to_limit'] == 1.5
    assert s['wrong_values'] == [(80, 50), (80, 70)]
