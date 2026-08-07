import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def detect_license_plates(model_path, image_path, output_image_path, output_json_path, conf):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(str(model_path), task="detect")
    result = model(image, optimize=True, verbose=False, conf=conf)[0]

    detections = []
    annotated_image = image.copy()

    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0]) if box.conf is not None else None
        class_id = int(box.cls[0]) if box.cls is not None else 0
        class_name = result.names.get(class_id, str(class_id))

        detections.append(
            {
                "index": index,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
            }
        )

        label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_image_path), annotated_image)
    output_json_path.write_text(
        json.dumps(detections, indent=2),
        encoding="utf-8",
    )

    return detections


def main():
    image_name = "4.jpg"
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "license_plate_detector_int8_openvino_model"
    image_path = base_dir / image_name
    output_image_path = base_dir / "plate_detected2.jpg"
    output_json_path = base_dir / "coordinates2.json"
    conf = 0.25

    detections = detect_license_plates(
        model_path=model_path,
        image_path=image_path,
        output_image_path=output_image_path,
        output_json_path=output_json_path,
        conf=conf,
    )

    if not detections:
        print("No license plate detected.")
        return

    print(f"Detected {len(detections)} license plate(s):")
    for detection in detections:
        coords = detection["coordinates"]
        print(
            f"- #{detection['index']} "
            f"conf={detection['confidence']:.3f} "
            f"x1={coords['x1']} y1={coords['y1']} "
            f"x2={coords['x2']} y2={coords['y2']} "
            f"width={coords['width']} height={coords['height']}"
        )
    print(f"Annotated image saved to: {output_image_path}")
    print(f"Coordinates saved to: {output_json_path}")


if __name__ == "__main__":
    main()
