"""Leitura e escrita simples de cfg Darknet com seções repetidas."""
from pathlib import Path


def read_cfg(path):
    blocks = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            blocks.append({"type": line.strip("[]")})
        else:
            key, value = line.split("=", 1)
            blocks[-1][key.strip()] = value.strip()
    return blocks


def write_cfg(blocks, path):
    text = "\n\n".join("\n".join([f'[{b["type"]}]'] +
                        [f"{k}={v}" for k, v in b.items() if k != "type"]) for b in blocks)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text + "\n")


def create_cfg(source, target, iterations=4000, size=320, batch=16, subdivisions=16):
    blocks = read_cfg(source)
    blocks[0].update(batch=str(batch), subdivisions=str(subdivisions), width=str(size), height=str(size),
                     max_batches=str(iterations), burn_in=str(min(100, max(1, iterations // 10))),
                     steps=f"{int(iterations*.8)},{int(iterations*.9)}")
    for index, block in enumerate(blocks):
        if block["type"] == "yolo":
            assert blocks[index - 1]["type"] == "convolutional"
            block.update(classes="82", random="0")
            blocks[index - 1]["filters"] = str(3 * (82 + 5))
    write_cfg(blocks, target)
    return blocks
