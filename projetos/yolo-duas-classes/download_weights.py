"""Baixa pesos e verifica o SHA-256 conhecido da cópia usada neste projeto."""
import hashlib
import json
import urllib.request
from pathlib import Path

SHA256 = "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c"
URLS = ["https://data.pjreddie.com/files/yolov3-tiny.weights",
        "https://raw.githubusercontent.com/mbaske/yolo-unity/master/YoloPretrained/YOLOv3-tiny/yolov3-tiny.weights"]


def download(output="weights/yolov3-tiny.weights"):
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        return target
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != SHA256:
                raise ValueError("SHA-256 diferente da cópia documentada")
            target.write_bytes(data)
            target.with_suffix(".source.json").write_text(json.dumps({"url": url, "sha256": SHA256}, indent=2))
            return target
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("Falha no download. Envie uma cópia com o hash documentado.\n" + "\n".join(errors))


if __name__ == "__main__":
    print(download())
