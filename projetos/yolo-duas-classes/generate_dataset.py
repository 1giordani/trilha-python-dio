"""Base sintética própria com duas classes novas; anotações geradas, não humanas."""
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
from annotations import convert_annotation, NEW_CLASSES


def generate(root="dataset", seed=42):
    root = Path(root)
    rng, manifest = random.Random(seed), []
    counts = {"train": 80, "valid": 20, "test": 20}
    for split, count in counts.items():
        paths = []
        for index in range(count):
            name = f"{split}_{index:03d}"
            image_path = root / "images" / split / f"{name}.png"
            xml_path = root / "annotations" / split / f"{name}.xml"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (256, 256), tuple(rng.randint(15, 65) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "filename").text = image_path.name
            for class_index, label in enumerate(NEW_CLASSES):
                size = rng.randint(42, 80)
                x = rng.randint(8 + 128 * class_index, 118 + 128 * class_index - size)
                y = rng.randint(10, 240 - size)
                color = tuple(rng.randint(130, 255) for _ in range(3))
                if class_index == 0:
                    points = [(x + size / 2, y), (x + size, y + size), (x, y + size)]
                else:
                    points = [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
                draw.polygon(points, fill=color)
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = label
                ET.SubElement(obj, "deleted").text = "0"
                polygon = ET.SubElement(obj, "polygon")
                for px, py in points:
                    pt = ET.SubElement(polygon, "pt")
                    ET.SubElement(pt, "x").text = str(px)
                    ET.SubElement(pt, "y").text = str(py)
            image.save(image_path)
            ET.ElementTree(annotation).write(xml_path, encoding="utf-8", xml_declaration=True)
            label_path = root / "labels" / split / f"{name}.txt"
            convert_annotation(xml_path, image_path, label_path)
            paths.append(str(image_path))
            manifest.append({"image": str(image_path), "split": split,
                             "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
        (root / f"{split}.txt").write_text("\n".join(paths) + "\n")
    assert len({item["sha256"] for item in manifest}) == len(manifest)
    with (root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "split", "sha256"])
        writer.writeheader()
        writer.writerows(manifest)
    report = {"origin": "imagens e anotações sintéticas próprias geradas por código; sem rotulagem humana",
              "seed": seed, "images": counts, "objects_per_image": 2, "classes": NEW_CLASSES,
              "identical_images_across_splits": False,
              "limitation": "triângulos à esquerda, quadrados à direita: viés espacial e domínio simples"}
    (root / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
