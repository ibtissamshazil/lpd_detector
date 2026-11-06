import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from ultralytics.engine.results import Results

from detect_plate_obb import bounding_box, detect_license_plates


class BoundingBoxTests(unittest.TestCase):
    def test_tilted_plate_enclosed_without_rounding_inward(self):
        points = np.array([[10.7, 20.2], [60.1, 10.8], [65.4, 30.5], [15.3, 40.9]])
        self.assertEqual(
            bounding_box(points, 100, 80),
            {"x1": 10, "y1": 10, "x2": 66, "y2": 41, "width": 56, "height": 31},
        )

    def test_clips_to_image_and_rejects_degenerate_boxes(self):
        points = np.array([[-5, -2], [120, 0], [110, 90], [0, 85]])
        self.assertEqual(
            bounding_box(points, 100, 80),
            {"x1": 0, "y1": 0, "x2": 100, "y2": 80, "width": 100, "height": 80},
        )
        self.assertIsNone(bounding_box(np.ones((4, 2)), 100, 80))
        self.assertIsNone(bounding_box(np.full((4, 2), np.nan), 100, 80))

    def test_empty_predictions_save_unchanged_image_and_empty_json(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        result = Results(image, path="input.png", names={0: "plate"},
                         obb=np.empty((0, 7), dtype=np.float32))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, coordinates = root / "input.png", root / "output.png", root / "out.json"
            cv2.imwrite(str(source), image)
            with patch("detect_plate_obb.YOLO") as model:
                model.return_value.predict.return_value = [result]
                self.assertEqual(detect_license_plates("model", source, output, coordinates), [])
                model.assert_called_once_with("model", task="obb")
            self.assertEqual(json.loads(coordinates.read_text()), [])
            np.testing.assert_array_equal(cv2.imread(str(output)), image)


if __name__ == "__main__":
    unittest.main()
