"""AP50 nas DUAS classes novas; não é COCO mAP@[.5:.95] nem avaliação COCO80."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from detector import DarknetDetector


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0


def average_precision(predictions, truths, threshold=0.5):
    used, tp, fp, points = set(), 0, 0, [(0, 1)]
    total = sum(len(boxes) for boxes in truths.values())
    if total == 0:
        return None
    for item in sorted(predictions, key=lambda p: -p["score"]):
        boxes = truths.get(item["image"], [])
        choices = [(iou(item["box"], box), j) for j, box in enumerate(boxes)]
        overlap, index = max(choices, default=(0, -1))
        key = (item["image"], index)
        if overlap >= threshold and key not in used:
            used.add(key)
            tp += 1
        else:
            fp += 1
        points.append((tp/total, tp/(tp+fp)))
    # Envelope de precisão e integral por incrementos de recall (all-points AP).
    points.append((1, 0))
    precision = [p for _, p in points]
    for j in range(len(precision)-2, -1, -1):
        precision[j] = max(precision[j], precision[j+1])
    return sum((points[j][0]-points[j-1][0])*precision[j]
               for j in range(1, len(points)) if points[j][0] > points[j-1][0])


def ground_truth(image_path):
    label = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
    with Image.open(image_path) as source:
        w, h = source.size
    result = []
    for line in label.read_text().splitlines():
        cls, x, y, bw, bh = map(float, line.split())
        result.append({"class_id": int(cls), "box": [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]})
    return result


def evaluate(darknet, cfg, weights, split="test", output="evidencias/avaliacao"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    detector = DarknetDetector(darknet, cfg, weights)
    images = Path(f"dataset/{split}.txt").read_text().splitlines()
    predictions, truths = [], {80: {}, 81: {}}
    for index, path in enumerate(images):
        gt = ground_truth(path)
        for class_id in truths:
            truths[class_id][path] = [x["box"] for x in gt if x["class_id"] == class_id]
        detected = detector.predict(path)
        predictions.extend(dict(d, image=path) for d in detected)
        if index == 0:
            with Image.open(path) as source:
                preview = source.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for d in detected:
                if d["score"] >= 0.25:
                    draw.rectangle(d["box"], outline="lime", width=2)
                    label = "triangulo" if d["class_id"] == 80 else "quadrado"
                    draw.text((d["box"][0], max(0, d["box"][1]-14)), f'{label} {d["score"]:.2f}',
                              fill="lime", font=ImageFont.load_default(size=12))
            preview.save(out / "predicao.png")
    ap = {str(cid): average_precision([p for p in predictions if p["class_id"] == cid], boxes)
          for cid, boxes in truths.items()}
    result = {"split": split, "images": len(images), "weights": Path(weights).name,
              "AP50_new_classes": ap, "mean_AP50_new_classes": sum(ap.values())/2,
              "iou_threshold": 0.5, "min_score": 0.01, "nms": 0.45,
              "preview_threshold": 0.25, "scope": "duas classes sintéticas; sem avaliação das 80 classes COCO"}
    (out / "metricas.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / "predicoes.json").write_text(json.dumps(predictions, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--darknet", required=True)
    p.add_argument("--cfg", default="config/yolov3-tiny-82.cfg")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--output", default="evidencias/avaliacao")
    a = p.parse_args()
    print(json.dumps(evaluate(a.darknet, a.cfg, a.weights, a.split, a.output), indent=2))
