# signd: speed limit signs from the camera

Adds a third speed limit source, `vision`, next to `car` (CAN) and `map` (OSM). It is meant for cars that don't report the posted limit on CAN.

```
road camera ──VisionIPC──> signd ──liveVisionSpeedLimitSP──> plannerd (SpeedLimitResolver) ──> longitudinalPlanSP ──> UI / SLA
                                  │
                   detector.py ─> sign_filter.py
          (sign detector + value classifier)   (needs several agreeing frames; holds the value until the next sign, a turn, 5 km or 10 min)
```

- **Setting:** `SpeedLimitVision` (Settings → Cruise → Speed Limit → Source).
  - `0` off.
  - `1` fallback: used only when the selected policy's sources have nothing.
  - `2` priority: tried before the selected policy's sources.
  - With `Combined`, the camera is one more input to the "lowest limit wins" rule.
- **Units:** sign values are read in the units set by `IsMetric` (km/h signs in Canada, mph in the US).
- **When signd runs:** onroad, with the setting on, and only when `models/sign_detector.onnx` and `models/sign_classifier.onnx` exist.
- **Scope (v1):** regulatory maximum-speed signs on the right side of the road. Advisory signs, "ahead" signs, school zones, minimum limits and truck limits must be classified as *reject*.

## Model contract

| model | input | output |
|---|---|---|
| `sign_detector.onnx` | `image` f32 `[1,3,640,640]`, RGB 0..1, letterboxed ROI (`detector.ROIS`) | `boxes` f32 `[N,6]` after NMS: `x1,y1,x2,y2` (input px), `score`, `class` (0 = speed limit sign) |
| `sign_classifier.onnx` | `crop` f32 `[B,3,64,64]`, RGB 0..1 | `logits` f32 `[B, 26]`: `SPEED_LIMIT_VALUES` (10..130 step 5), then reject |

- **Runtimes:** onnxruntime runs the models when it is installed (PC/Mac). Otherwise tinygrad's `OnnxRunner` runs them, as on the device.
- **Licensing:** keep the training code and weights MIT/Apache compatible.
- **Test fixtures:** `tests/fake_models.py` writes constant-output models that follow this contract. They are used by the tests and are handy for pipeline checks.

## Offline workflow (no driving)

Copy routes off the device, from `/data/media/0/realdata/<route>--<n>/` (`fcamera.hevc` is the road camera signd uses, `ecamera.hevc` the wide camera, plus `rlog`/`qlog`).

```bash
source .venv/bin/activate

# 1. collect training images: ROI crops at 1 Hz
python -m openpilot.sunnypilot.selfdrive.signd.tools.sign_eval ~/drives/<route> --dump-rois rois/ --rate 1

# 2. label the posted limit over frame ranges in labels.csv
#    route,segment,start_frame,end_frame,limit   (20 Hz frame index within the segment)

# 3. evaluate a model: timeline.json, score.json, annotated mp4s
python -m openpilot.sunnypilot.selfdrive.signd.tools.sign_eval ~/drives/<route> \
  --models path/to/models --labels labels.csv --out eval/ --video
```

How to read `score.json`:
- `precision`: when a limit is shown, how often it is right. Optimise this first; a wrong limit is worse than none.
- `coverage`: the share of labeled time with the right limit shown.
- `median_time_to_limit`: how long after a limit starts before it is shown.

## Tests

```bash
python tools/test_runner.py openpilot/sunnypilot/selfdrive/signd openpilot/sunnypilot/selfdrive/controls/lib/speed_limit
```
