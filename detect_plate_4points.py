import json
from pathlib import Path

import cv2
import numpy as np

from detect_plate import detect_license_plates


def order_points(points):
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def predict_plate_corners(image, coordinates):
    image_height, image_width = image.shape[:2]
    box_width = coordinates["x2"] - coordinates["x1"]
    box_height = coordinates["y2"] - coordinates["y1"]
    padding_x = max(2, round(box_width * 0.06))
    padding_y = max(2, round(box_height * 0.12))

    roi_x1 = max(0, coordinates["x1"] - padding_x)
    roi_y1 = max(0, coordinates["y1"] - padding_y)
    roi_x2 = min(image_width, coordinates["x2"] + padding_x)
    roi_y2 = min(image_height, coordinates["y2"] + padding_y)
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    minimum_area = box_width * box_height * 0.10
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        center, size, angle = cv2.minAreaRect(contour)
        rect_width, rect_height = size
        if rect_width == 0 or rect_height == 0:
            continue

        aspect_ratio = max(size) / min(size)
        fill_ratio = area / (rect_width * rect_height)
        centered_x = abs(center[0] - roi.shape[1] / 2) < roi.shape[1] * 0.35
        centered_y = abs(center[1] - roi.shape[0] / 2) < roi.shape[0] * 0.35
        if (
            area >= minimum_area
            and 2 <= aspect_ratio <= 7
            and fill_ratio >= 0.55
            and centered_x
            and centered_y
        ):
            candidates.append((area, contour, (center, size, angle)))

    if not candidates:
        return np.array(
            [
                [coordinates["x1"], coordinates["y1"]],
                [coordinates["x2"], coordinates["y1"]],
                [coordinates["x2"], coordinates["y2"]],
                [coordinates["x1"], coordinates["y2"]],
            ],
            dtype=np.float32,
        )

    _, plate_contour, plate_rectangle = max(candidates, key=lambda item: item[0])
    hull = cv2.convexHull(plate_contour)
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(polygon) == 4 and cv2.isContourConvex(polygon):
        corners = polygon.reshape(4, 2).astype(np.float32)
    else:
        corners = cv2.boxPoints(plate_rectangle)
    corners += np.array([roi_x1, roi_y1], dtype=np.float32)
    return order_points(corners)


def straighten_plate(image, corners):
    top_left, top_right, bottom_right, bottom_left = order_points(corners)
    width = round(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = round(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    width = max(width, 1)
    height = max(height, 1)

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_and_straighten_plates(
    model_path,
    image_path,
    output_image_path,
    output_json_path,
    output_4points_json_path,
    output_crop_path,
    conf,
):
    detections = detect_license_plates(
        model_path=model_path,
        image_path=image_path,
        output_image_path=output_image_path,
        output_json_path=output_json_path,
        conf=conf,
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    four_point_detections = []
    for detection in detections:
        corners = predict_plate_corners(image, detection["coordinates"])
        straightened_plate = straighten_plate(image, corners)
        crop_path = output_crop_path.with_stem(
            f"{output_crop_path.stem}_{detection['index']}"
        )
        cv2.imwrite(str(crop_path), straightened_plate)

        corner_names = ("top_left", "top_right", "bottom_right", "bottom_left")
        corner_coordinates = {
            name: {"x": int(round(point[0])), "y": int(round(point[1]))}
            for name, point in zip(corner_names, corners)
        }
        four_point_detections.append(
            {
                "index": detection["index"],
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "straightened_image": crop_path.name,
                "coordinates": corner_coordinates,
                "width": straightened_plate.shape[1],
                "height": straightened_plate.shape[0],
            }
        )

    output_4points_json_path.write_text(
        json.dumps(four_point_detections, indent=2),
        encoding="utf-8",
    )
    return four_point_detections


def main():
    base_dir = Path(__file__).resolve().parent
    detections = detect_and_straighten_plates(
        model_path=base_dir / "license_plate_detector_int8_openvino_model",
        image_path=base_dir / "14.jpg",
        output_image_path=base_dir / "plate_detected3.jpg",
        output_json_path=base_dir / "coordinates2.json",
        output_4points_json_path=base_dir / "coordinates_4points.json",
        output_crop_path=base_dir / "plate_straightened2.jpg",
        conf=0.25,
    )

    if not detections:
        print("No license plate detected.")
        return

    for detection in detections:
        print(
            f"Straightened plate #{detection['index']} saved to: "
            f"{base_dir / detection['straightened_image']}"
        )
    print(f"Four-point coordinates saved to: {base_dir / 'coordinates_4points.json'}")


if __name__ == "__main__":
    main()
