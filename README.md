# License plate bounding boxes

`detect_plate_obb.py` uses the supplied `best_int8_openvino_model` through
Ultralytics' YOLO26 OBB implementation and OpenVINO CPU inference.
It encloses the model's four rotated corners in a simple upright rectangle;
it does not estimate corners using contours or straighten the plate.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe detect_plate_obb.py
.venv/Scripts/python.exe detect_plate_obb.py 14.jpg --conf 0.25
```

Default input: `3.jpg`, matching the car in `plate_detected2.jpg`.
Outputs beside the input: `<stem>_plate_obb.jpg` (green boxes and confidence)
and `<stem>_plate_obb.json` (existing index/class/confidence/coordinates schema).
Coordinates use original-image pixels, clipped to image bounds, with
`x1`, `y1`, `x2`, `y2`, `width`, `height`; crop using `image[y1:y2, x1:x2]`.
No detections produces an unchanged image and `[]` JSON.
Repeated runs overwrite those new output filenames.

Requires Ultralytics >=8.4.90 for the YOLO26 end-to-end OBB export.
Existing detector/cropping scripts remain separate.

```powershell
.venv/Scripts/python.exe -m unittest test_detect_plate_obb -v
```

Local validation detected plates in `3.jpg`, `4.jpg`, `5.jpg`, `14.jpg`,
and `14a.jpg`; `15.jpg` produced no detections even at confidence 0.05.
These examples do not establish general detection accuracy.

## Compare a downloaded dataset

```powershell
.venv/Scripts/python.exe compare_dataset.py path/to/links.txt
```

Input: one HTTPS image link per line; duplicates removed in input order.
Default output: `dataset/` (`--dataset PATH` overrides it).
Six download workers retry failures up to three attempts and reuse readable
cached images. `dataset/links.txt` and `manifest.json` preserve the local input
list and download outcomes, including source URLs; keep these local.

Both models load once, running CPU inference at 640 and the same confidence
threshold (default 0.25, configurable with `--conf`). Existing detector functions
accept an optional preloaded `model`; their standalone entry points still work.
The old detector uses `license_plate_detector_int8_openvino_model`; the new
detector uses `best_int8_openvino_model`, enclosing rotated predictions in
upright rectangles. Model-specific postprocessing remains unchanged.

Open `dataset/comparison.html`: black, basic, side-by-side old/new annotations,
confidence, counts, original-image links, and explicit download/inference errors.
Category dropdown filters All, Both detect, Old only, New only, Neither,
Count mismatch, and Errors, with category totals and a visible-row count.
Count mismatch overlaps detection categories; error rows belong only to Errors.
Filtering runs locally in the browser without downloads or inference.
All unique inputs receive a row, including failures; images load lazily.
`old/` and `new/` contain image/coordinate outputs; `results.json` records results.
Reruns reuse downloads and regenerate inference outputs/report. Keep the entire
dataset folder together for the local HTML's relative image links to work.
Prediction counts do not measure accuracy without ground-truth labels.

Completed 500-image comparison: all downloads and both inference passes succeeded.
Old model detected plates in 490 images (517 boxes); new model in 499 images
(514 boxes). Verified all 500 report rows and 1,000 annotated images/JSON outputs.

```powershell
.venv/Scripts/python.exe -m unittest test_compare_dataset test_detect_plate_obb -v
```
