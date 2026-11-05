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
