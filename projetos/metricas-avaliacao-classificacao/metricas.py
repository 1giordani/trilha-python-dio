"""Métricas binárias implementadas pelas fórmulas, sem biblioteca de ML."""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrizConfusao:
    vp: int
    vn: int
    fp: int
    fn: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} deve ser inteiro não negativo.")

    @property
    def total(self):
        return self.vp + self.vn + self.fp + self.fn

    def como_matriz(self):
        """Linhas: real; colunas: previsto. Ordem: negativo, positivo."""
        return [[self.vn, self.fp], [self.fn, self.vp]]


def dividir(numerador, denominador):
    # None explicita uma métrica indefinida, sem fabricar um resultado zero.
    return numerador / denominador if denominador else None


def acuracia(matriz):
    return dividir(matriz.vp + matriz.vn, matriz.total)


def sensibilidade(matriz):
    return dividir(matriz.vp, matriz.vp + matriz.fn)


def especificidade(matriz):
    return dividir(matriz.vn, matriz.vn + matriz.fp)


def precisao(matriz):
    return dividir(matriz.vp, matriz.vp + matriz.fp)


def f_score(matriz):
    # F-score aqui é F1: média harmônica com pesos iguais para precisão e recall.
    # A forma por contagens evita intermediários indefinidos quando VP=0.
    return dividir(2 * matriz.vp, 2 * matriz.vp + matriz.fp + matriz.fn)


def calcular_metricas(matriz):
    return {"acuracia": acuracia(matriz), "sensibilidade_recall": sensibilidade(matriz),
            "especificidade": especificidade(matriz), "precisao": precisao(matriz),
            "f1_score": f_score(matriz)}


def percentual(valor):
    return "indefinida (denominador zero)" if valor is None else f"{100 * valor:.2f}%"


def gerar_relatorio(matriz, output_dir="resultados"):
    metrics = calcular_metricas(matriz)
    result = {"tipo": "classificação binária", "contagens": asdict(matriz),
              "total": matriz.total, "matriz": matriz.como_matriz(),
              "ordem": {"linhas": "real: negativo, positivo", "colunas": "previsto: negativo, positivo"},
              "metricas": metrics,
              "divisao_por_zero": "None no Python; null no JSON; indefinida no relatório"}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metricas.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "\n".join(f"| {name} | {percentual(value)} |" for name, value in metrics.items())
    text = f"""# Avaliação da matriz de confusão

Total: {matriz.total}. Classe positiva e origem dos dados devem ser definidas pelo experimento.

| Real / Previsto | Negativo | Positivo |
| --- | --- | --- |
| Negativo | VN = {matriz.vn} | FP = {matriz.fp} |
| Positivo | FN = {matriz.fn} | VP = {matriz.vp} |

| Métrica | Resultado |
| --- | --- |
{table}

Os valores do JSON mantêm a precisão dos cálculos; a exibição é arredondada para duas casas percentuais.
Uma métrica indefinida não equivale a zero. F-score representa F1 neste projeto.
"""
    (output / "relatorio.md").write_text(text, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vp", type=int, default=40)
    parser.add_argument("--vn", type=int, default=45)
    parser.add_argument("--fp", type=int, default=10)
    parser.add_argument("--fn", type=int, default=5)
    parser.add_argument("--saida", default="resultados")
    args = parser.parse_args()
    try:
        matrix = MatrizConfusao(args.vp, args.vn, args.fp, args.fn)
        result = gerar_relatorio(matrix, args.saida)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print("Matriz: linhas reais e colunas previstas; ordem negativo, positivo")
    print(matrix.como_matriz())
    print(f"Total: {matrix.total}")
    for name, value in result["metricas"].items():
        print(f"{name}: {percentual(value)}")


if __name__ == "__main__":
    main()
