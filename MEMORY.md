# Project context

- Python/OpenCV/Ultralytics/OpenVINO plate detection and crop scripts.
- `best_int8_openvino_model`: YOLO26n-OBB, INT8, 640x640, end2end,
  one plate class; export metadata version 8.4.90, task `obb`.
- New entry point `detect_plate_obb.py`: native Ultralytics OBB decoding,
  CPU inference, four corners -> enclosing axis-aligned rectangle.
- JSON retains existing detector schema; integer bounds floor/ceil then clip;
  x2/y2 exclusive for crops. Default `3.jpg` matches reference car.
- Separate `<stem>_plate_obb.jpg` and `.json` outputs; repeated runs overwrite.
- Existing scripts and reference outputs had user modifications; preserve them.
- Requirements now Ultralytics >=8.4.90,<9; local environment upgraded to 8.4.183.
- Geometry checks cover tilted boxes, clipping, invalid/degenerate boxes;
  empty-result integration check covers image/JSON persistence.
- Real inference: five supplied images detect plates, confidence ~0.94-0.98;
  `15.jpg` misses even at 0.05. No heuristic fallback; accuracy not established.
- Sandbox may block default Ultralytics settings path; during checks set
  YOLO_CONFIG_DIR to a writable local directory.
- Approved dataset comparison: `compare_dataset.py links.txt`, default `dataset/`.
  Six download threads, three attempts, validated local cache, hashed filenames;
  links/manifest remain local. Optional preloaded model in both detector functions.
- Both models run CPU/640/conf 0.25; old YOLOv8 detect vs new YOLO26 OBB.
  One model load per pass; errors retained per image; outputs old/new, results.json,
  black basic comparison.html with lazy side-by-side images and confidence/counts.
- Comparison checks: input validation/deduplication, escaped HTML, every-row/error
  coverage; no ground truth, counts must not be described as accuracy.
- Completed approved 500-image run: 500 downloads, zero download/inference errors.
  Old: 490 images with detections / 517 boxes; new: 499 images / 514 boxes.
  Verified 500 HTML rows, 1,000 readable annotated images and matching JSON;
  seven unit tests pass, pip check and diff whitespace check pass.
  Browser preview unavailable: CUA sandbox helper initialization failed.
- Comparison HTML now has local category dropdown: all/both/old-only/new-only/
  neither/count mismatch/errors, totals, visible count, empty-state message.
  Mismatch overlaps valid detection categories; errors isolated. Existing HTML
  regenerated from results.json without inference; eight tests pass and Node
  executes actual report JavaScript against all seven categories.
