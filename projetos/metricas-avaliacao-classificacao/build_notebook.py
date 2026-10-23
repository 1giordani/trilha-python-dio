"""Gera notebook independente com as mesmas funções de metricas.py."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source,
                  "execution_count": None, "outputs": []})


md("""# Cálculo de métricas de classificação — DIO
**Autor:** Giordani Martins Silva. Código e documentação preparados com auxílio de IA.

Objetivo: implementar acurácia, sensibilidade (recall), especificidade, precisão e F-score pelas fórmulas. Usamos uma matriz binária arbitrária, como permite o desafio; não foi treinado um modelo nem utilizada uma base real.

Execute as células em ordem. Não requer GPU, downloads ou bibliotecas de aprendizado de máquina. O notebook usa somente a biblioteca padrão do Python.
""")
md("""## 1. Definir a classe positiva e a matriz
Exemplo fictício: **positivo = peça com defeito**, negativo = peça sem defeito.

| Real / Previsto | Negativo | Positivo |
| --- | --- | --- |
| Negativo | VN = 45 | FP = 10 |
| Positivo | FN = 5 | VP = 40 |

São 100 peças fictícias. VP e VN são acertos; FP são alarmes falsos; FN são defeitos não detectados. A ordem das linhas e colunas é sempre negativo, positivo. Não são dados de uma empresa.
""")
md("""## 2. Fórmulas
Com N = VP + VN + FP + FN:

| Métrica | Fórmula | Pergunta |
| --- | --- | --- |
| Acurácia | (VP + VN) / N | Qual fração de todas as previsões está correta? |
| Sensibilidade / recall | VP / (VP + FN) | Dos positivos reais, quantos foram detectados? |
| Especificidade | VN / (VN + FP) | Dos negativos reais, quantos foram reconhecidos? |
| Precisão | VP / (VP + FP) | Dos positivos previstos, quantos eram positivos reais? |
| F1-score | 2 × P × S / (P + S) | Como equilibrar precisão e sensibilidade? |

O F-score escolhido é F1, com pesos iguais. Pelo cálculo direto das contagens: F1 = 2VP / (2VP + FP + FN). Essa forma permite F1=0 quando há erro e VP=0, mesmo se uma das métricas intermediárias não for definida. Se o próprio denominador for zero, retornamos None, não zero.
""")
source = (root / "metricas.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
code(source)
md("## 3. Calcular e exportar\nAltere as quatro contagens abaixo para testar outra matriz. Contagens negativas ou não inteiras são rejeitadas.")
code('''matriz = MatrizConfusao(vp=40, vn=45, fp=10, fn=5)
resultado = gerar_relatorio(matriz, "resultados")
print("Matriz (real nas linhas; previsto nas colunas):", matriz.como_matriz())
print("Total:", matriz.total)
for nome, valor in resultado["metricas"].items():
    print(f"{nome}: {percentual(valor)}")
''')
md("## 4. Conferir um cálculo manual\nPara a matriz padrão: 85/100 = 85% de acurácia; 40/45 de sensibilidade; 45/55 de especificidade; 40/50 de precisão; 80/95 de F1. A checagem abaixo usa uma matriz fixa própria, sem impedir que você altere a anterior.")
code('''from math import isclose
exemplo = calcular_metricas(MatrizConfusao(40, 45, 10, 5))
esperado = {"acuracia": 85/100, "sensibilidade_recall": 40/45,
            "especificidade": 45/55, "precisao": 40/50, "f1_score": 80/95}
assert all(isclose(exemplo[nome], valor) for nome, valor in esperado.items())
print("Cinco cálculos conferidos.")
''')
md("## 5. Entender denominadores zero\nSem positivos previstos, precisão é indefinida. Sem positivos reais, recall é indefinido. Sem negativos reais, especificidade é indefinida. Matriz vazia não permite calcular acurácia. None vira null no JSON.")
code('''casos = {
    "matriz vazia": MatrizConfusao(0, 0, 0, 0),
    "sem positivos previstos": MatrizConfusao(0, 10, 0, 5),
    "sem positivos reais": MatrizConfusao(0, 10, 5, 0),
}
for nome, caso in casos.items():
    print(nome, calcular_metricas(caso))
''')
md("## 6. Consultar os arquivos gerados\nO JSON preserva os valores decimais calculados, e o Markdown exibe percentuais com duas casas. No Colab, os arquivos estão no painel lateral da pasta resultados.")
code('''print(Path("resultados/relatorio.md").read_text(encoding="utf-8"))
print("JSON:", Path("resultados/metricas.json").resolve())
''')
md("""## Reflexão
- Precisão e sensibilidade têm denominadores diferentes: confundi-las muda a interpretação.
- Muitos FN reduzem o recall; muitos FP reduzem a precisão e a especificidade.
- Acurácia elevada pode esconder falhas na classe rara. Analise a matriz e várias métricas.
- F1 não usa VN diretamente; a classe positiva escolhida deve estar explícita.
- Este exercício calcula métricas de uma matriz arbitrária, não comprova desempenho de um modelo real.

O notebook é publicado sem saídas para execução limpa. As evidências locais estão no repositório; não foi realizada uma sessão no Colab durante a preparação.
""")
for index, cell in enumerate(cells):
    cell["id"] = f"metricas-{index:02d}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "colab": {"name": "metricas_avaliacao.ipynb"}}}
(root / "metricas_avaliacao.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook gerado: {len(cells)} células")
