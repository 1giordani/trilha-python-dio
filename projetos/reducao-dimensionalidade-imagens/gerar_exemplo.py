"""Gera uma imagem de teste própria: gradiente RGB e áreas de cores conhecidas."""
from pathlib import Path
from PIL import Image, ImageDraw


def criar_exemplo(path="exemplos/entrada.png"):
    width, height = 384, 256
    image = Image.new("RGB", (width, height))
    image.putdata([(round(255 * x / (width - 1)), round(255 * y / (height - 1)),
                    round(255 * (1 - x / (width - 1))))
                   for y in range(height) for x in range(width)])
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(["red", "lime", "blue", "yellow", "cyan", "magenta", "black", "white"]):
        draw.rectangle((index * 48, 192, (index + 1) * 48 - 1, 255), fill=color)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


if __name__ == "__main__":
    print(criar_exemplo())
