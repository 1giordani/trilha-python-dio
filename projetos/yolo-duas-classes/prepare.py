"""Prepara dataset e cfg de 82 classes; execute a partir da pasta do projeto."""
import argparse
from pathlib import Path
from darknet_config import create_cfg
from generate_dataset import generate


def prepare(darknet, iterations=150, size=128, batch=4):
    darknet = Path(darknet)
    Path("config").mkdir(exist_ok=True)
    Path("backup").mkdir(exist_ok=True)
    names = (darknet / "data/coco.names").read_text().strip().splitlines()
    assert len(names) == 80
    Path("config/classes.names").write_text("\n".join(names + ["marcador_triangulo", "marcador_quadrado"]) + "\n")
    create_cfg(darknet / "cfg/yolov3-tiny.cfg", "config/yolov3-tiny-82.cfg",
               iterations, size, batch, batch)
    Path("config/dataset.data").write_text(
        "classes=82\ntrain=dataset/train.txt\nvalid=dataset/valid.txt\n"
        "names=config/classes.names\nbackup=backup\n")
    return generate()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--darknet", required=True)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    if a.iterations < 1 or a.size < 32 or a.size % 32 or a.batch < 1:
        p.error("Use iterações e batch positivos e tamanho múltiplo de 32")
    print(prepare(a.darknet, a.iterations, a.size, a.batch))
