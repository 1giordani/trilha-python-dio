"""Conversão explícita por pixel: RGB -> cinza -> binária (0 ou 255)."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps


def preparar_rgb(image):
    """Respeita orientação EXIF e compõe transparência sobre fundo branco."""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def converter_cinza(image):
    """Luma aproximada: 0,299R + 0,587G + 0,114B, arredondada para inteiro."""
    if image.mode != "RGB":
        raise ValueError("converter_cinza requer uma imagem RGB; use preparar_rgb.")
    result = Image.new("L", image.size)
    pixels = image.load()
    values = []
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]
            values.append((299 * r + 587 * g + 114 * b + 500) // 1000)
    result.putdata(values)
    return result


def binarizar(gray, limiar=128):
    """Cinza >= limiar vira branco (255); abaixo vira preto (0)."""
    if type(limiar) is not int or not 0 <= limiar <= 255:
        raise ValueError("O limiar deve ser um inteiro entre 0 e 255.")
    if gray.mode != "L":
        raise ValueError("binarizar requer uma imagem em modo L.")
    result = Image.new("L", gray.size)
    result.putdata([255 if value >= limiar else 0 for value in gray.tobytes()])
    return result


def montar_comparacao(rgb, gray, binary, limiar):
    from PIL import ImageDraw, ImageFont
    width, height = rgb.size
    margin, title_height = 12, 42
    canvas = Image.new("RGB", (3 * width + 4 * margin, height + title_height + margin), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    titles = ["Original RGB", "Cinza: 0 a 255", f"Binaria: limiar {limiar}"]
    for index, (image, title) in enumerate(zip([rgb, gray, binary], titles)):
        x = margin + index * (width + margin)
        draw.text((x, 12), title, fill="#171717", font=font)
        canvas.paste(image.convert("RGB"), (x, title_height))
    return canvas


def processar(input_path, output_dir="resultados", limiar=128):
    output = Path(output_dir)
    input_path = Path(input_path)
    # Valida o limiar antes de ler ou criar arquivos.
    if type(limiar) is not int or not 0 <= limiar <= 255:
        raise ValueError("O limiar deve ser um inteiro entre 0 e 255.")
    with Image.open(input_path) as source:
        rgb = preparar_rgb(source)
    gray = converter_cinza(rgb)
    binary = binarizar(gray, limiar)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ["original.png", "cinza.png", "binaria.png", "comparacao.png", "relatorio.json"]]
    if input_path.resolve() in [p.resolve() for p in targets]:
        raise ValueError("A saída sobrescreveria a entrada; escolha outra pasta.")
    rgb.save(output / "original.png")
    gray.save(output / "cinza.png")
    binary.save(output / "binaria.png")
    montar_comparacao(rgb, gray, binary, limiar).save(output / "comparacao.png")
    values = list(binary.tobytes())
    report = {
        "entrada": input_path.name, "largura": rgb.width, "altura": rgb.height,
        "pixels": rgb.width * rgb.height, "limiar": limiar,
        "formula": "(299*R + 587*G + 114*B + 500) // 1000",
        "regra": "cinza >= limiar: 255; caso contrário: 0",
        "modos": {"original": rgb.mode, "cinza": gray.mode, "binaria": binary.mode},
        "intervalo_cinza": list(gray.getextrema()),
        "valores_binaria": sorted(set(values)),
        "pixels_pretos": values.count(0), "pixels_brancos": values.count(255),
        "dimensoes_preservadas": rgb.size == gray.size == binary.size,
    }
    (output / "relatorio.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imagem", help="Caminho de uma imagem PNG ou JPEG")
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--limiar", type=int, default=128)
    args = parser.parse_args()
    try:
        report = processar(args.imagem, args.saida, args.limiar)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
