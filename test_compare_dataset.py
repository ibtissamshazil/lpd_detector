import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from compare_dataset import download_image, read_links, render_report


class ComparisonTests(unittest.TestCase):
    def test_filter_categories_include_overlap_and_isolate_errors(self):
        def row(identifier, old_count, new_count):
            return {"id": identifier, "download": "ok", **{
                key: {"image": f"{key}/{identifier}.jpg", "detections": [{"confidence": 0.8}] * count}
                for key, count in (("old", old_count), ("new", new_count))}}
        rows = [row("1", 1, 1), row("2", 1, 0), row("3", 0, 1),
                row("4", 0, 0), row("5", 1, 2), row("6", 0, 0)]
        rows[-1]["new"] = {"error": "failed"}
        report = render_report(rows, 0.25)
        for identifier, categories in (("1", "both"), ("2", "old-only mismatch"),
                                       ("3", "new-only mismatch"), ("4", "neither"),
                                       ("5", "both mismatch"), ("6", "errors")):
            self.assertIn(f'id="image-{identifier}" data-categories="{categories}"', report)
        self.assertIn('value="mismatch">Count mismatch (3)', report)
        self.assertIn('value="errors">Errors (1)', report)
        self.assertIn('id="visible-count"', report)

    def test_links_deduplicated_in_input_order_and_invalid_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "links.txt"
            path.write_text("https://example.com/b.jpg\n\nhttps://example.com/a.jpg\nhttps://example.com/b.jpg\n")
            self.assertEqual(read_links(path), ["https://example.com/b.jpg", "https://example.com/a.jpg"])
            path.write_text("file:///secret.jpg")
            with self.assertRaises(ValueError):
                read_links(path)

    def test_report_includes_every_row_errors_and_escaped_content(self):
        rows = [
            {"id": "0001", "download": "ok", "old": {"detections": [], "image": "old/0001.jpg"},
             "new": {"detections": [{"confidence": 0.9}], "image": "new/0001.jpg"}},
            {"id": "0002", "download": "error", "error": "<bad>"},
        ]
        report = render_report(rows, 0.25)
        self.assertEqual(report.count('<article '), 2)
        self.assertIn("&lt;bad&gt;", report)
        self.assertNotIn("<bad>", report)
        self.assertIn("0.900", report)
        self.assertIn('loading="lazy"', report)
        self.assertIn("No plates", report)

    def test_download_retry_validation_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = {"id": "0001", "url": "https://example.com/image.jpg", "source": "image.jpg"}
            _, encoded = cv2.imencode(".jpg", np.zeros((12, 20, 3), dtype=np.uint8))
            response = MagicMock()
            response.__enter__.return_value.read.return_value = encoded.tobytes()
            with patch("compare_dataset.urlopen", side_effect=[OSError("transient"), response]) as fetch:
                with patch("compare_dataset.time.sleep"):
                    self.assertEqual(download_image((row, root))["download"], "ok")
                self.assertEqual(fetch.call_count, 2)
            with patch("compare_dataset.urlopen") as fetch:
                self.assertEqual(download_image((row, root))["download"], "cached")
                fetch.assert_not_called()

    def test_invalid_download_recorded_as_error_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = {"id": "0001", "url": "https://example.com/image.jpg", "source": "image.jpg"}
            response = MagicMock()
            response.__enter__.return_value.read.return_value = b"not an image"
            with patch("compare_dataset.urlopen", return_value=response) as fetch:
                with patch("compare_dataset.time.sleep"):
                    result = download_image((row, root))
                self.assertEqual(result["download"], "error")
                self.assertEqual(fetch.call_count, 3)
            self.assertFalse((root / "image.jpg").exists())


if __name__ == "__main__":
    unittest.main()
