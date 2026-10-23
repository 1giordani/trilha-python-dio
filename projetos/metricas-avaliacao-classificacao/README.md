# Cálculo de métricas de avaliação de aprendizado

Projeto DIO em Python que implementa **acurácia, sensibilidade (recall), especificidade, precisão e F1-score**, usando as fórmulas de uma matriz de confusão binária. O cálculo utiliza somente a biblioteca padrão, sem funções prontas de métricas de bibliotecas de aprendizado de máquina.

**Autor:** Giordani Martins Silva. Implementação e documentação preparadas com auxílio de IA.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1giordani/trilha-python-dio/blob/main/projetos/metricas-avaliacao-classificacao/metricas_avaliacao.ipynb)

## Matriz escolhida

O enunciado permite escolher uma matriz arbitrária. O exemplo é uma inspeção **fictícia** de 100 peças: classe positiva = peça com defeito; classe negativa = peça sem defeito. Não são dados da Vale nem de outra empresa. Não há treinamento ou base de dados real neste exercício.

As linhas representam o valor real; as colunas, a previsão. A ordem é negativo, positivo.

| Real / Previsto | Negativo | Positivo | Total real |
| --- | --- | --- | --- |
| Negativo | VN = 45 | FP = 10 | 55 |
| Positivo | FN = 5 | VP = 40 | 45 |
| Total previsto | 50 | 50 | 100 |

- VP: defeito corretamente detectado.
- VN: peça sem defeito corretamente reconhecida.
- FP: peça sem defeito indicada como defeituosa; alarme falso.
- FN: peça defeituosa indicada como sem defeito; falha de detecção.

No código, a matriz é representada por [[VN, FP], [FN, VP]]. Os parâmetros são nomeados para evitar trocar sua posição.

## Fórmulas e resultados executados

P representa precisão; S representa sensibilidade. N = VP + VN + FP + FN.

| Métrica | Fórmula | Substituição | Resultado |
| --- | --- | --- | --- |
| Acurácia | (VP + VN) / N | 85 / 100 | 85,00% |
| Sensibilidade / recall | VP / (VP + FN) | 40 / 45 | 88,89% |
| Especificidade | VN / (VN + FP) | 45 / 55 | 81,82% |
| Precisão | VP / (VP + FP) | 40 / 50 | 80,00% |
| F1-score | 2PS / (P + S) | 80 / 95, pela forma equivalente abaixo | 84,21% |

O F-score implementado é **F1**, com pesos iguais para precisão e recall. A fórmula equivalente usada no código é 2VP / (2VP + FP + FN). Não se trata de média aritmética. As funções retornam proporções entre 0 e 1; somente a apresentação multiplica por 100. O JSON não arredonda as métricas para duas casas antes de salvá-las.

Os resultados acima foram calculados pela implementação e conferidos por testes. Veja o [JSON executado](exemplos/metricas.json) e o [relatório gerado](exemplos/relatorio.md).

## Executar

Use Python 3.10 ou superior. Não é necessário instalar pacotes:

```bash
git clone https://github.com/1giordani/trilha-python-dio.git
cd trilha-python-dio/projetos/metricas-avaliacao-classificacao
python metricas.py
```

Para informar outra matriz:

```bash
python metricas.py --vp 40 --vn 45 --fp 10 --fn 5 --saida resultados
```

Saída padrão:

```text
Matriz: linhas reais e colunas previstas; ordem negativo, positivo
[[45, 10], [5, 40]]
Total: 100
acuracia: 85.00%
sensibilidade_recall: 88.89%
especificidade: 81.82%
precisao: 80.00%
f1_score: 84.21%
```

A pasta indicada recebe metricas.json e relatorio.md. Reexecutar na mesma pasta substitui esses dois resultados. Também é possível importar MatrizConfusao e as cinco funções de metricas.py sem executar a interface de terminal.

## Executar no Colab

Clique no botão acima e execute as células na ordem. O notebook é autossuficiente, não precisa de GPU nem de download de dados. Altere as contagens na célula da matriz para comparar cenários. O script build_notebook.py regenera o notebook a partir das mesmas funções de metricas.py.

## Divisão por zero e validação

As contagens devem ser inteiros não negativos. Valores negativos, fracionários, booleanos e strings são rejeitados pela classe; a CLI aceita somente texto que represente um inteiro válido.

Quando o denominador de uma métrica é zero, a função retorna **None**. No JSON isso é null e, no relatório, “indefinida (denominador zero)”. Essa escolha evita confundir ausência de informação com um resultado zero.

| Situação | Consequência |
| --- | --- |
| N = 0 | Acurácia indefinida |
| VP + FN = 0 | Sensibilidade indefinida; não existem positivos reais |
| VN + FP = 0 | Especificidade indefinida; não existem negativos reais |
| VP + FP = 0 | Precisão indefinida; não existem positivos previstos |
| 2VP + FP + FN = 0 | F1 indefinido |

Se VP=0 e FP+FN>0, o F1 é 0 pela fórmula direta das contagens, mesmo quando precisão ou sensibilidade isoladamente não podem ser calculadas. A convenção está explícita e é testada. Outras bibliotecas podem oferecer convenções diferentes para divisões por zero.

## Testes e evidências

```bash
python -m unittest test_metricas -v
python validar_notebook.py
```

Os testes cobrem cálculo manual, classificação perfeita, erros em todas as previsões, matriz vazia, ausência de positivos reais/previstos ou de negativos reais, equivalência da fórmula harmônica do F1, orientação da matriz, contagens inválidas e exportação JSON com null.

Veja [registro dos testes](evidencias/testes.txt), [execução do exemplo](evidencias/execucao.txt) e [validação das células](evidencias/validacao.json). O notebook foi executado como células Python localmente, em uma pasta temporária. Não houve sessão no Colab nem treinamento de modelo. Aulas devem ser assistidas pelo estudante.

## Interpretação

No cenário fictício, 5 de 45 defeitos foram perdidos: recall de 88,89%. Dos 50 alarmes de defeito, 40 estavam corretos: precisão de 80%. Entre as 55 peças sem defeito, 45 foram reconhecidas: especificidade de 81,82%.

A acurácia considera todos os acertos; em bases desbalanceadas, pode ocultar falhas na classe rara. F1 resume precisão e recall, mas não usa VN diretamente. A escolha da classe positiva e o custo de FP e FN devem acompanhar a leitura das métricas. O exemplo explica cálculos, sem demonstrar desempenho de um classificador real.

## Arquivos

- metricas.py: validação das contagens, funções, CLI e relatórios.
- test_metricas.py: testes automatizados.
- metricas_avaliacao.ipynb: notebook para Colab/Jupyter.
- build_notebook.py: geração do notebook.
- validar_notebook.py: checagem de sintaxe, sincronização e execução local das células.
- exemplos/: JSON e Markdown gerados com a matriz escolhida.
- evidencias/: resultados reais das verificações.
- ENTREGA-DIO.md: link e descrição para submissão.

Proposta educacional: Digital Innovation One, desafio “Cálculo de Métricas de Avaliação de Aprendizado”. A imagem da tabela mencionada no enunciado não foi fornecida nesta mensagem; o projeto documenta as fórmulas usuais das cinco métricas solicitadas, incluindo a definição adotada de F-score como F1.
