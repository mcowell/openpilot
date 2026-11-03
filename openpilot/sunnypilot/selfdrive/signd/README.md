# signd: speed limit signs from the camera

Adds a third speed limit source, `vision`, next to `car` (CAN) and `map` (OSM). It is meant for cars that don't report the posted limit on CAN.

```
road camera ──VisionIPC──> signd ──liveVisionSpeedLimitSP──> plannerd (SpeedLimitResolver) ──> longitudinalPlanSP ──> UI / SLA
                                  │
                   detector.py ─> sign_filter.py
          (sign detector + value classifier)   (needs several agreeing frames; holds the value until the next sign, an end-of-zone sign, a turn, 5 km or 10 min)
```

- **Setting:** `SpeedLimitVision` (Settings → Cruise → Speed Limit → Source).
  - `0` off.
  - `1` fallback: used only when the selected policy's sources have nothing.
  - `2` priority: tried before the selected policy's sources.
  - With `Combined`, the camera is one more input to the "lowest limit wins" rule.
- **Units:** sign values are read in the units set by `IsMetric` (km/h signs in Canada, mph in the US).
- **When signd runs:** onroad, with the setting on, and only when the detector and classifier exist (on device, compiled by `SConscript`).
- **Scope (v1):** regulatory maximum-speed signs on the right side of the road. Advisory signs, "ahead" signs, school zones, minimum limits and truck limits must be classified as *reject*.
- **End of zone:** signs with an "ENDS / FIN" tab are classified as *end*. Once confirmed, signd drops the held limit and reports none, so the resolver falls back to the other sources.
- **Scheduling (`reader.py`):** all GPU work is split into small jobs (~10-15 ms on a 3X), each run just after the driving model finishes a frame, in the gap before the next one, so signd never holds the GPU when modeld needs it. A long full-ROI scan (~80 ms) made modeld drop frames. Jobs: *tile* (the 160 px local detector on one of 8 overlapping tiles; the two "entry" tiles, where signs first appear, every other job), *local* (re-find a tracked sign in a window around its predicted position) and *classify* (a crop cut on the CPU from the frame the sign was found in). One job every other frame, every frame while a sign is tracked: signs are often readable for only ~1 s. Over all drives this matches a fixed 2 Hz full scan (precision 81% vs 77%, 18 vs 17 limits found) at ~140 ms of GPU per second in <15 ms pieces. Without the local detector signd falls back to the full scan at 2 Hz. `liveVisionSpeedLimitSP` is published at 2 Hz (`SIGND_RATE_HZ`).
- **Priority:** signd runs at the lowest GPU priority and on the little CPU cores, so the driving model preempts it.
- **Guard (`guard.py`):** selfdrived soft-disables (modeldLagging) when `modelV2.frameDropPerc` > 1, and one dropped frame already reads ~0.5%. On any new dropped frame signd pauses detector work (5 s, doubling up to 60 s if drops recur), then runs scans only until 20 s pass without drops.

## Model contract

| model | input | output |
|---|---|---|
| `sign_detector.onnx` | `image` f32 `[1,3,H,W]`, RGB 0..1, the ROI (`detector.ROIS`) at half resolution, letterboxed (416x416 today) | raw YOLO `[1,4+C,A]` (decoded and NMS'd in `detector.py`) or `boxes` f32 `[N,6]` after NMS: `x1,y1,x2,y2` (input px), `score`, `class` (0 = speed limit sign) |
| `sign_detector_local.onnx` (optional) | same as the detector, 160x160: a window around a tracked sign, sized so the panel is ~30 px | same as the detector. Without it signd scans at 2 Hz and doesn't track |

- **3X GPU compiler:** the Adreno shader compiler crashes ("back-end lowering failure") on this YOLO11n at most input sizes with openpilot's flags. 416 and 160 compile; 192-384 (step 32), 448 and 640 don't. Check a new size or architecture on a device before shipping it: a model that doesn't compile breaks the build.
| `sign_classifier.onnx` | `crop` f32 `[1,3,96,64]`, RGB 0..1: `classifier_crop()`, the panel plus the tabs below it | `logits` f32 `[1, 27]`: `SPEED_LIMIT_VALUES` (10..130 step 5), reject, end of zone |

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
