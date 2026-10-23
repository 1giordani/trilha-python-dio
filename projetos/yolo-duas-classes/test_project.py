import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from annotations import normalized_box, convert_annotation
from darknet_config import create_cfg, read_cfg
from evaluate import iou, average_precision
from generate_dataset import generate

ROOT = Path(__file__).resolve().parent


class ProjectTests(unittest.TestCase):
    def test_normalizacao(self):
        self.assertEqual(normalized_box([(10, 20), (30, 60)], 100, 100), [.2, .4, .2, .4])

    def test_caixas_invalidas(self):
        for points in [[(1, 1), (1, 3)], [(-1, 2), (3, 4)], [(0, 0), (101, 1)]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                normalized_box(points, 100, 100)

    def test_labelme_desktop_json(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            Image.new("RGB", (100, 100)).save(d / "i.png")
            (d / "i.json").write_text(json.dumps({"shapes": [{"label": "marcador_quadrado",
                "shape_type": "rectangle", "points": [[10,20],[30,60]]}]}))
            result = convert_annotation(d / "i.json", d / "i.png", d / "i.txt")
            self.assertEqual(result, ["81 0.20000000 0.40000000 0.20000000 0.40000000"])

    def test_labelme_xml_e_separacao_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            report = generate(Path(directory)/"dataset")
            self.assertEqual(report["images"], {"train":80,"valid":20,"test":20})
            files = list((Path(directory)/"dataset/labels").rglob("*.txt"))
            self.assertEqual(len(files), 120)
            for file in files:
                rows = [line.split() for line in file.read_text().splitlines()]
                self.assertEqual({int(row[0]) for row in rows}, {80, 81})
                self.assertTrue(all(0 < float(v) <= 1 for row in rows for v in row[1:]))

    def test_cfg_classes_filtros(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"new.cfg"
            create_cfg(ROOT/"referencias/yolov3-tiny-80.cfg", path, 150, 128, 4, 4)
            blocks = read_cfg(path)
            heads = [i for i,b in enumerate(blocks) if b["type"] == "yolo"]
            self.assertEqual(len(heads), 2)
            for index in heads:
                self.assertEqual(blocks[index]["classes"], "82")
                self.assertEqual(blocks[index-1]["filters"], "261")

    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]), 1)
        self.assertEqual(iou([0,0,1,1],[2,2,3,3]), 0)

    def test_ap_perfeito_e_sem_deteccao(self):
        truth = {"a": [[0,0,10,10]]}
        self.assertEqual(average_precision([{"image":"a","score":.9,"box":[0,0,10,10]}], truth), 1)
        self.assertEqual(average_precision([], truth), 0)
        self.assertIsNone(average_precision([], {}))

    def test_ap_falso_positivo_antes_de_acerto(self):
        pred = [{"image":"a","score":.9,"box":[20,20,30,30]},
                {"image":"a","score":.8,"box":[0,0,10,10]}]
        self.assertEqual(average_precision(pred, {"a":[[0,0,10,10]]}), .5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
