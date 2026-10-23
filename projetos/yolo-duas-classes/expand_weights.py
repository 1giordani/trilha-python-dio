"""Transplanta pesos YOLOv3-tiny COCO80 para saídas COCO80 + duas classes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from darknet_config import read_cfg


def convolution_layout(blocks):
    channels, layout = [], []
    previous = int(blocks[0].get("channels", 3))
    for index, block in enumerate(blocks[1:]):
        kind = block["type"]
        if kind == "convolutional":
            filters = int(block["filters"])
            size = int(block["size"])
            groups = int(block.get("groups", 1))
            layout.append((index, filters, previous // groups, size,
                           int(block.get("batch_normalize", 0))))
            previous = filters
        elif kind == "route":
            refs = [int(x) for x in block["layers"].split(",")]
            previous = sum(channels[index + ref if ref < 0 else ref] for ref in refs)
        elif kind not in ("maxpool", "upsample", "yolo"):
            raise ValueError(f"Camada não suportada pelo conversor tiny: {kind}")
        channels.append(previous)
    return layout


def read_weights(path, blocks):
    with Path(path).open("rb") as stream:
        version = np.fromfile(stream, dtype="<i4", count=3)
        if len(version) != 3 or np.any(version < 0):
            raise ValueError("Cabeçalho inválido")
        wide = int(version[0]) * 10 + int(version[1]) >= 2
        seen = np.fromfile(stream, dtype="<u8" if wide else "<u4", count=1)
        if len(seen) != 1:
            raise ValueError("Cabeçalho incompleto")
        data = np.fromfile(stream, dtype="<f4")
    offset, result = 0, {}
    for index, filters, inputs, size, bn in convolution_layout(blocks):
        count = filters * (4 if bn else 1)
        if offset + count > len(data):
            raise ValueError("Pesos incompletos")
        vectors = data[offset:offset + count].reshape(4 if bn else 1, filters).copy()
        offset += count
        count = filters * inputs * size * size
        if offset + count > len(data):
            raise ValueError("Pesos incompletos")
        kernels = data[offset:offset + count].reshape(filters, inputs, size, size).copy()
        result[index] = (vectors, kernels)
        offset += count
    if offset != len(data) or not np.isfinite(data).all():
        raise ValueError("Tamanho ou valores dos pesos não correspondem ao cfg")
    return version, result


def expand(source_cfg, target_cfg, input_weights, output_weights):
    old, new = read_cfg(source_cfg), read_cfg(target_cfg)
    version, tensors = read_weights(input_weights, old)
    rng, checked = np.random.default_rng(42), 0
    heads = {i - 1 for i, b in enumerate(old[1:]) if b["type"] == "yolo"}
    output = Path(output_weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        version.astype("<i4").tofile(stream)
        np.array([0], dtype="<u8" if int(version[0])*10+int(version[1]) >= 2 else "<u4").tofile(stream)
        for index, filters, inputs, size, bn in convolution_layout(new):
            vectors, kernels = tensors[index]
            if index in heads:
                if vectors.shape != (1, 255) or filters != 261 or bn:
                    raise ValueError("Esperadas cabeças tiny COCO80 -> 82")
                old_v = vectors.reshape(1, 3, 85)
                old_k = kernels.reshape(3, 85, inputs, size, size)
                v = np.zeros((1, 3, 87), dtype="<f4")
                k = rng.normal(0, 0.01, (3, 87, inputs, size, size)).astype("<f4")
                # Por âncora: bbox(4), objectness(1), 80 classes antigas, 2 novas.
                v[:, :, :85], k[:, :85] = old_v, old_k
                v[:, :, 85:] = -4.6
                assert np.array_equal(v[:, :, :85], old_v)
                assert np.array_equal(k[:, :85], old_k)
                vectors, kernels = v.reshape(1, filters), k.reshape(filters, inputs, size, size)
                checked += 1
            elif kernels.shape != (filters, inputs, size, size):
                raise ValueError("Arquiteturas incompatíveis")
            vectors.astype("<f4").tofile(stream)
            kernels.astype("<f4").tofile(stream)
    if checked != 2:
        raise ValueError("Esperadas duas cabeças de detecção")
    # Também verifica se o arquivo produzido é legível e sem bytes excedentes.
    _, restored = read_weights(output, new)
    for index in heads:
        v, k = restored[index]
        ov, ok = tensors[index]
        assert np.array_equal(v.reshape(1, 3, 87)[:, :, :85], ov.reshape(1, 3, 85))
        assert np.array_equal(k.reshape(3, 87, *k.shape[1:])[:, :85], ok.reshape(3, 85, *ok.shape[1:]))
    report = {"old_classes": 80, "new_classes": 82, "heads_expanded": checked,
              "old_outputs_copied_exactly": True, "seen_reset": True,
              "input_sha256": hashlib.sha256(Path(input_weights).read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_cfg")
    parser.add_argument("target_cfg")
    parser.add_argument("weights")
    parser.add_argument("output")
    a = parser.parse_args()
    print(json.dumps(expand(a.source_cfg, a.target_cfg, a.weights, a.output), indent=2))
