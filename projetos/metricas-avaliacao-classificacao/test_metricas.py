import json
import tempfile
import unittest
from pathlib import Path
from metricas import MatrizConfusao, calcular_metricas, gerar_relatorio, percentual


class MetricasTests(unittest.TestCase):
    def test_exemplo_calculado_manualmente(self):
        values = calcular_metricas(MatrizConfusao(40, 45, 10, 5))
        expected = {"acuracia": 85 / 100, "sensibilidade_recall": 40 / 45,
                    "especificidade": 45 / 55, "precisao": 40 / 50, "f1_score": 80 / 95}
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(values[name], value)

    def test_classificacao_perfeita(self):
        self.assertTrue(all(v == 1 for v in calcular_metricas(MatrizConfusao(7, 9, 0, 0)).values()))

    def test_todas_previsoes_erradas(self):
        self.assertTrue(all(v == 0 for v in calcular_metricas(MatrizConfusao(0, 0, 7, 9)).values()))

    def test_matriz_vazia(self):
        self.assertTrue(all(v is None for v in calcular_metricas(MatrizConfusao(0, 0, 0, 0)).values()))

    def test_sem_positivos_previstos(self):
        values = calcular_metricas(MatrizConfusao(0, 10, 0, 5))
        self.assertIsNone(values["precisao"])
        self.assertEqual(values["sensibilidade_recall"], 0)
        self.assertEqual(values["f1_score"], 0)

    def test_sem_positivos_reais(self):
        values = calcular_metricas(MatrizConfusao(0, 10, 5, 0))
        self.assertIsNone(values["sensibilidade_recall"])
        self.assertEqual(values["precisao"], 0)
        self.assertEqual(values["f1_score"], 0)

    def test_sem_negativos_reais(self):
        self.assertIsNone(calcular_metricas(MatrizConfusao(10, 0, 0, 5))["especificidade"])

    def test_f1_equivale_media_harmonica(self):
        values = calcular_metricas(MatrizConfusao(11, 19, 4, 7))
        p, s = values["precisao"], values["sensibilidade_recall"]
        self.assertAlmostEqual(values["f1_score"], 2 * p * s / (p + s))

    def test_orientacao_matriz(self):
        self.assertEqual(MatrizConfusao(40, 45, 10, 5).como_matriz(), [[45, 10], [5, 40]])

    def test_rejeita_contagens_invalidas(self):
        for field in ["vp", "vn", "fp", "fn"]:
            for invalid in [-1, 1.5, "3", True]:
                data = dict(vp=1, vn=1, fp=1, fn=1)
                data[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    MatrizConfusao(**data)

    def test_exportacao_e_null(self):
        with tempfile.TemporaryDirectory() as directory:
            gerar_relatorio(MatrizConfusao(0, 10, 0, 5), directory)
            data = json.loads((Path(directory) / "metricas.json").read_text())
            self.assertIsNone(data["metricas"]["precisao"])
            self.assertTrue((Path(directory) / "relatorio.md").is_file())
            self.assertEqual(percentual(0.85), "85.00%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
