"""LabelMe web XML / Labelme desktop JSON -> caixas YOLO normalizadas."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

NEW_CLASSES = {"marcador_triangulo": 80, "marcador_quadrado": 81}


def normalized_box(points, width, height):
    if width <= 0 or height <= 0 or len(points) < 2:
        raise ValueError("Dimensões ou pontos insuficientes")
    if not all(math.isfinite(float(v)) for point in points for v in point):
        raise ValueError("Coordenadas não finitas")
    xs, ys = zip(*points)
    if min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height:
        raise ValueError("Anotação fora da imagem; revise-a no LabelMe")
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Caixa sem área")
    return [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]


def convert_annotation(annotation, image, output):
    annotation = Path(annotation)
    with Image.open(image) as source:
        width, height = source.size
    objects = []
    if annotation.suffix.lower() == ".xml":
        root = ET.parse(annotation).getroot()
        for obj in root.findall("object"):
            if obj.findtext("deleted", "0") == "1":
                continue
            label = obj.findtext("name", "")
            points = [(float(p.findtext("x")), float(p.findtext("y"))) for p in obj.findall("polygon/pt")]
            objects.append((label, points))
    elif annotation.suffix.lower() == ".json":
        data = json.loads(annotation.read_text())
        for shape in data["shapes"]:
            if shape.get("shape_type", "polygon") not in ("rectangle", "polygon"):
                raise ValueError("Use retângulo ou polígono; pontos/linhas/círculos não são suportados")
            objects.append((shape["label"], shape["points"]))
    else:
        raise ValueError("Use XML do LabelMe web ou JSON do Labelme desktop")
    rows = []
    for label, points in objects:
        if label not in NEW_CLASSES:
            raise ValueError(f"Classe desconhecida: {label}")
        box = normalized_box(points, width, height)
        rows.append(str(NEW_CLASSES[label]) + " " + " ".join(f"{v:.8f}" for v in box))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""))
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("annotation")
    p.add_argument("image")
    p.add_argument("output")
    a = p.parse_args()
    print(convert_annotation(a.annotation, a.image, a.output))
