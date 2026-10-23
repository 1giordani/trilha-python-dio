"""Processa imagem local com várias faces cadastradas, sem serviço de rede."""
import argparse
import json
from pathlib import Path
from PIL import Image
from faces import FaceRecognizer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="artefatos/classificador.keras")
    p.add_argument("--metadata", default="artefatos/metadata.json")
    p.add_argument("--output", default="resultados")
    a = p.parse_args()
    if Path(a.image).resolve() == (Path(a.output)/"reconhecimento.png").resolve():
        p.error("A saída sobrescreveria a entrada; escolha outra pasta.")
    system = FaceRecognizer(a.model, a.metadata)
    with Image.open(a.image) as source:
        preview, detections = system.recognize(source)
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    preview.save(out/"reconhecimento.png")
    (out/"deteccoes.json").write_text(json.dumps(detections,ensure_ascii=False,indent=2))
    print(json.dumps(detections,ensure_ascii=False,indent=2))
