import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from processar import preparar_rgb, converter_cinza, binarizar, processar


class ConversaoTests(unittest.TestCase):
    def test_cores_conhecidas(self):
        image = Image.new("RGB", (5, 1))
        image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)])
        self.assertEqual(list(converter_cinza(image).tobytes()), [76, 150, 29, 0, 255])

    def test_borda_do_limiar(self):
        image = Image.new("L", (4, 1))
        image.putdata([0, 127, 128, 255])
        self.assertEqual(list(binarizar(image).tobytes()), [0, 0, 255, 255])

    def test_todos_os_niveis(self):
        image = Image.new("RGB", (256, 1))
        image.putdata([(i, i, i) for i in range(256)])
        gray = converter_cinza(image)
        self.assertEqual(list(gray.tobytes()), list(range(256)))
        self.assertEqual(set(binarizar(gray).tobytes()), {0, 255})

    def test_limiares_extremos(self):
        image = Image.new("L", (3, 1))
        image.putdata([0, 128, 255])
        self.assertEqual(list(binarizar(image, 0).tobytes()), [255, 255, 255])
        self.assertEqual(list(binarizar(image, 255).tobytes()), [0, 0, 255])

    def test_rejeita_limiar_invalido(self):
        for value in [-1, 256, 128.0, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                binarizar(Image.new("L", (1, 1)), value)

    def test_transparencia_sobre_branco(self):
        rgb = preparar_rgb(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        self.assertEqual(rgb.getpixel((0, 0)), (255, 255, 255))

    def test_preserva_entrada_e_dimensoes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "entrada.png"
            Image.new("RGB", (7, 3), (15, 200, 80)).save(source)
            original = source.read_bytes()
            out = Path(directory) / "saida"
            report = processar(source, out)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(report["dimensoes_preservadas"])
            self.assertEqual(report["pixels_pretos"] + report["pixels_brancos"], 21)
            for name in ["cinza.png", "binaria.png"]:
                with Image.open(out / name) as image:
                    self.assertEqual(image.size, (7, 3))
                    self.assertEqual(image.mode, "L")
            self.assertEqual(json.loads((out / "relatorio.json").read_text())["pixels"], 21)

    def test_bloqueia_sobrescrever_entrada(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.png"
            Image.new("RGB", (1, 1), "red").save(source)
            with self.assertRaises(ValueError):
                processar(source, directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
