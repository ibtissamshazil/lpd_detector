import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def bounding_box(points, image_width, image_height):
    """Enclose all four OBB corners; coordinates also support NumPy crops."""
    if not np.isfinite(points).all():
        return None
    x1, y1 = np.floor(points.min(axis=0)).astype(int)
    x2, y2 = np.ceil(points.max(axis=0)).astype(int)
    x1, x2 = np.clip([x1, x2], 0, image_width).tolist()
    y1, y2 = np.clip([y1, y2], 0, image_height).tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": x2 - x1, "height": y2 - y1,
    }


def detect_license_plates(model_path, image_path, output_image_path, output_json_path, conf=0.25, model=None):
    if not 0 <= conf <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if model is None:
        model = YOLO(str(model_path), task="obb")
    result = model.predict(image, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
    if result.obb is None:
        raise ValueError("Model must produce oriented bounding boxes (task='obb').")

    detections = []
    annotated_image = image.copy()
    for box in result.obb.cpu():
        points = box.xyxyxyxy[0].numpy()
        coordinates = bounding_box(points, image.shape[1], image.shape[0])
        if coordinates is None:
            continue
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names.get(class_id, str(class_id))
        detections.append({
            "index": len(detections),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "coordinates": coordinates,
        })
        x1, y1 = coordinates["x1"], coordinates["y1"]
        x2, y2 = coordinates["x2"], coordinates["y2"]
        cv2.rectangle(annotated_image, (x1, y1),
                      (min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)),
                      (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{class_name} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if not cv2.imwrite(str(output_image_path), annotated_image):
        raise OSError(f"Could not write image: {output_image_path}")
    Path(output_json_path).write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return detections


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Detect simple plate boxes using YOLO26 OBB/OpenVINO.")
    parser.add_argument("image", nargs="?", type=Path, default=base_dir / "5.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    output_image = args.image.with_name(f"{args.image.stem}_plate_obb.jpg")
    output_json = args.image.with_name(f"{args.image.stem}_plate_obb.json")
    detections = detect_license_plates(
        base_dir / "best_int8_openvino_model", args.image, output_image, output_json, args.conf,
    )
    print(f"Detected {len(detections)} license plate(s).")
    print(f"Annotated image: {output_image}")
    print(f"Coordinates: {output_json}")


if __name__ == "__main__":
    main()
