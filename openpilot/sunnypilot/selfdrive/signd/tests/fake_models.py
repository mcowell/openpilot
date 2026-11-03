"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Constant-output ONNX models that follow the detector.py contract, to exercise the pipeline before
a trained model exists:  python -m openpilot.sunnypilot.selfdrive.signd.tests.fake_models <dir> [value]
"""
import sys
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.selfdrive.signd import SPEED_LIMIT_VALUES
from openpilot.sunnypilot.selfdrive.signd.detector import CLS_SIZE, DET_SIZE


def write_fake_models(out_dir: Path, value: int = 80, box=(400., 200., 440., 250.), score: float = 0.9) -> None:
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  out_dir.mkdir(parents=True, exist_ok=True)

  # detector: image -> one fixed box (the input is reduced and multiplied by 0 so it stays in the graph)
  boxes = numpy_helper.from_array(np.array([[*box, score, 0.]], dtype=np.float32), 'const_boxes')
  graph = helper.make_graph(
    [helper.make_node('ReduceMax', ['image'], ['m'], keepdims=0),
     helper.make_node('Mul', ['m', 'zero'], ['z']),
     helper.make_node('Add', ['const_boxes', 'z'], ['boxes'])],
    'fake_sign_detector',
    [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, DET_SIZE, DET_SIZE])],
    [helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6])],
    [boxes, numpy_helper.from_array(np.array(0., dtype=np.float32), 'zero')])
  onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8), out_dir / 'sign_detector.onnx')

  # classifier: crop -> logits peaked at `value`, one row per crop
  logits = np.full((1, len(SPEED_LIMIT_VALUES) + 1), -5., dtype=np.float32)
  logits[0, SPEED_LIMIT_VALUES.index(value)] = 5.
  graph = helper.make_graph(
    [helper.make_node('ReduceMean', ['crop'], ['m'], axes=[1, 2, 3], keepdims=0),
     helper.make_node('Unsqueeze', ['m', 'axis1'], ['m1']),
     helper.make_node('Mul', ['m1', 'zero'], ['z']),
     helper.make_node('Add', ['z', 'const_logits'], ['logits'])],
    'fake_sign_classifier',
    [helper.make_tensor_value_info('crop', TensorProto.FLOAT, ['B', 3, CLS_SIZE, CLS_SIZE])],
    [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['B', len(SPEED_LIMIT_VALUES) + 1])],
    [numpy_helper.from_array(logits, 'const_logits'), numpy_helper.from_array(np.array(0., dtype=np.float32), 'zero'),
     numpy_helper.from_array(np.array([1], dtype=np.int64), 'axis1')])
  onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8), out_dir / 'sign_classifier.onnx')


if __name__ == '__main__':
  write_fake_models(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 80)
