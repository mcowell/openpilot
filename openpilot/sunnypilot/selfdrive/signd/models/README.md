# signd models (proof of concept)

- `sign_detector.onnx`: YOLO11n single-class speed limit sign detector, trained with Ultralytics (AGPL-3.0).
  Training data includes Mapillary imagery (CC BY-SA 4.0). Not upstream-clean; to be replaced by a clean detector.
- `sign_classifier.onnx`: value classifier (10..130 km/h step 5 + reject), trained on synthetic signs,
  own drives and Mapillary crops (CC BY-SA 4.0).
