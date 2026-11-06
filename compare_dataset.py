"""Download image links and compare both local plate detectors offline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np


def read_links(path):
    links = list(dict.fromkeys(line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
                               if line.strip()))
    if not links:
        raise ValueError("Link file is empty.")
    for link in links:
        parsed = urlsplit(link)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Each nonempty line must be an HTTPS image URL.")
    return links


def download_image(item):
    row, root = item
    destination = root / row["source"]
    if destination.exists() and cv2.imread(str(destination)) is not None:
        return {**row, "download": "cached"}
    for attempt in range(3):
        try:
            request = Request(row["url"], headers={"User-Agent": "PlateComparison/1.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read()
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Downloaded content is not a readable image.")
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return {**row, "download": "ok"}
        except Exception as error:
            if attempt == 2:
                return {**row, "download": "error", "error": str(error)}
            time.sleep(attempt + 1)


def render_report(rows, conf):
    escape = html.escape
    downloaded = sum(row["download"] != "error" for row in rows)
    totals = {}
    for key in ("old", "new"):
        successful = [row[key] for row in rows if key in row and "error" not in row[key]]
        totals[key] = (sum(bool(result["detections"]) for result in successful),
                       sum(len(result["detections"]) for result in successful), len(successful))

    def panel(row, key, title):
        result = row.get(key, {})
        if row["download"] == "error":
            body = f'<p class="error">Download failed: {escape(row["error"])}</p>'
        elif "error" in result:
            body = f'<p class="error">Inference failed: {escape(result["error"])}</p>'
        else:
            scores = ", ".join(f'{box["confidence"]:.3f}' for box in result["detections"])
            detail = f'{len(result["detections"])} plate(s) | confidence {scores}' if scores else "No plates"
            body = (f'<p>{detail}</p><a href="{escape(result["image"], quote=True)}">'
                    f'<img loading="lazy" src="{escape(result["image"], quote=True)}" alt="{title}"></a>')
        return f'<section><h3>{title}</h3>{body}</section>'

    filters = {"all": "All", "both": "Both detect", "old-only": "Old only",
               "new-only": "New only", "neither": "Neither",
               "mismatch": "Count mismatch", "errors": "Errors"}
    counts = dict.fromkeys(filters, 0)
    counts["all"] = len(rows)
    articles = []
    for row in rows:
        if row["download"] == "error" or any("error" in row.get(key, {}) for key in ("old", "new")):
            categories = ["errors"]
        else:
            old_count = len(row["old"]["detections"])
            new_count = len(row["new"]["detections"])
            categories = ["both" if old_count and new_count else
                          "old-only" if old_count else "new-only" if new_count else "neither"]
            if old_count != new_count:
                categories.append("mismatch")
        for category in categories:
            counts[category] += 1
        original = (f' · <a href="{escape(row["source"], quote=True)}">Original</a>'
                    if row["download"] != "error" and "source" in row else "")
        articles.append(f'<article id="image-{row["id"]}" data-categories="{" ".join(categories)}"><h2>Image {row["id"]}{original}</h2><div class="pair">'
                        + panel(row, "old", "detect_plate · old model")
                        + panel(row, "new", "detect_plate_obb · new model") + '</div></article>')
    summary = " | ".join(f'{key}: {values[0]} images with plates, {values[1]} boxes, {values[2]} processed'
                         for key, values in totals.items())
    options = "".join(f'<option value="{key}">{label} ({counts[key]})</option>'
                      for key, label in filters.items())
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plate detection comparison</title>
<style>
body{{background:#080808;color:#ddd;font:14px Arial,sans-serif;margin:20px}}
a{{color:#aaa}} h1{{font-size:22px}} h2{{font-size:16px}} h3{{font-size:14px;font-weight:normal}}
header{{border-bottom:1px solid #444;padding-bottom:16px}} article{{border-bottom:1px solid #333;padding:16px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} section{{min-width:0}}
img{{display:block;width:100%;height:420px;object-fit:contain;background:#000}}
.error{{color:#ff9b9b}} p{{overflow-wrap:anywhere}}
select{{background:#111;color:#ddd;border:1px solid #555;padding:8px;margin-right:12px}}
@media(max-width:650px){{body{{margin:8px}}.pair{{gap:6px}}img{{height:240px}}}}
</style><header><h1>Plate detection comparison · {len(rows)} images</h1>
<p>{downloaded} downloaded/cached · {len(rows) - downloaded} download failures · confidence ≥ {conf} · CPU · 640 input</p>
<p>Left: license_plate_detector_int8_openvino_model · Right: best_int8_openvino_model</p>
<p>{summary}</p><p>Counts describe predictions, not accuracy; no ground-truth labels. Click images to inspect full size.</p>
<label for="category">Category </label><select id="category">{options}</select>
<span id="visible-count" role="status" aria-live="polite">Showing {len(rows)} of {len(rows)}</span>
<p id="empty-filter" hidden>No images in this category.</p>
</header>{''.join(articles)}
<script>
const category = document.getElementById('category');
const articles = Array.from(document.querySelectorAll('article[data-categories]'));
function applyFilter() {{
    let visible = 0;
    for (const article of articles) {{
        article.hidden = category.value !== 'all' && !article.dataset.categories.split(' ').includes(category.value);
        if (!article.hidden) visible++;
    }}
    document.getElementById('visible-count').textContent = `Showing ${{visible}} of ${{articles.length}}`;
    document.getElementById('empty-filter').hidden = visible !== 0;
}}
category.addEventListener('change', applyFilter);
applyFilter();
</script></html>'''


def run(links_path, root, conf):
    if not 0 <= conf <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    links = read_links(links_path)
    root = Path(root).resolve()
    for directory in ("images", "old", "new"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    # Keep inputs local so reruns do not depend on the original attachment.
    (root / "links.txt").write_text("\n".join(links) + "\n", encoding="utf-8")
    rows = []
    for index, link in enumerate(links, 1):
        identifier = f"{index:04d}"
        digest = hashlib.sha256(link.encode()).hexdigest()[:12]
        rows.append({"id": identifier, "url": link, "source": f"images/{identifier}_{digest}.jpg"})
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = []
        for row in pool.map(download_image, ((row, root) for row in rows)):
            downloaded.append(row)
            if len(downloaded) % 25 == 0 or len(downloaded) == len(rows):
                print(f"Download {len(downloaded)}/{len(rows)}", flush=True)
    rows = downloaded
    (root / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    base = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(base / "Ultralytics"))
    from ultralytics import YOLO
    from detect_plate import detect_license_plates as detect_old
    from detect_plate_obb import detect_license_plates as detect_new

    specifications = (("old", "license_plate_detector_int8_openvino_model", "detect", detect_old),
                      ("new", "best_int8_openvino_model", "obb", detect_new))
    for key, folder, task, detector in specifications:
        model = YOLO(str(base / folder), task=task)
        for index, row in enumerate(rows, 1):
            if row["download"] == "error":
                continue
            image_path = f'{key}/{row["id"]}.jpg'
            json_path = f'{key}/{row["id"]}.json'
            try:
                detections = detector(base / folder, root / row["source"], root / image_path,
                                      root / json_path, conf, model=model)
                if not (root / image_path).is_file() or cv2.imread(str(root / image_path)) is None:
                    raise OSError("Annotated image was not written.")
                row[key] = {"image": image_path, "json": json_path, "detections": detections}
            except Exception as error:
                row[key] = {"error": str(error)}
            if index % 25 == 0 or index == len(rows):
                print(f"{key} inference {index}/{len(rows)}", flush=True)
        (root / "results.json").write_text(json.dumps({"conf": conf, "rows": rows}, indent=2), encoding="utf-8")
    report = root / "comparison.html"
    report.write_text(render_report(rows, conf), encoding="utf-8")
    print(f"Report: {report}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("links", type=Path, help="Text file containing one HTTPS image URL per line")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    run(args.links, args.dataset, args.conf)


if __name__ == "__main__":
    main()
