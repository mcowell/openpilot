# signd models (proof of concept)

- `sign_detector.onnx`: YOLO11n single-class speed limit sign detector at 416x416, trained with Ultralytics (AGPL-3.0).
  Training data includes Mapillary imagery (CC BY-SA 4.0). Not upstream-clean; to be replaced by a clean detector.
- `sign_detector_local.onnx`: the same detector weights exported at 160x160, for re-finding a tracked sign.
- `sign_classifier.onnx`: value classifier on 64x96 panel+tab crops (10..130 km/h step 5, reject, end of zone), trained on synthetic signs,
  own drives and Mapillary crops (CC BY-SA 4.0).

Negatives (classifier v7, 2026-11-04): licence plates were read as signs at red lights, and a Highway 416 crown shield
as 40. The classifier now also trains on licence plates found in own drives while stopped or slow (OWLv2, reviewed by
eye), plates signd actually misread, and synthetic King's Highway shields, all as 'reject'. On held-out drives
(incl. the 2026-11-04 drive): 35/36 signs read (was 33), no plate, shield or end-of-zone accepts (was one plate). A
detector retrained with the same plates was tried and not shipped: it found more signs but read a dusk "50 AREA ENDS"
as 50 and lost another sign.

Classifier v9 (2026-11-10): retrained with signs from the 2026-11-08/09 drives (416, city, dusk), reviewed by eye.
Yellow advisory signs, "ahead" signs (up arrow), green guide signs and route shields on those drives are trained as
'reject'. Crops OCR could not read are no longer trained as 'reject' (v8 did, and learned to reject real dusk signs).
Every accepted limit over all 28 drives was checked by eye: 200 correct (v7: 179), on held-out drives 58 (v7: 52) of
64 signs found by any version; the only wrong accept is an "ahead" 80 on a Highway 416 off-ramp, which v7 also makes
(v7 additionally accepted an "ahead" 60 and a county road shield as 15). Detector unchanged.
