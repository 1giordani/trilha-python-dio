# Resultados produzidos pelo notebook ou train.py

Após uma execução completa, esta pasta conterá:

| Arquivo | Conteúdo |
| --- | --- |
| metricas.json | Configuração, versões, splits, comparação de validação, métricas do teste, distribuição das classes e matriz |
| historico.json | Histórico das duas fases de treinamento |
| curvas.png | Acurácia e perda de treino e validação por época |
| matriz_confusao.png | Classes verdadeiras nas linhas e previstas nas colunas |
| amostras.png | Exemplos do conjunto de treino |
| predicoes.csv | Rótulo verdadeiro, probabilidade da classe 1 e rótulo previsto |
| classes.json | Ordem das classes usada na inferência |
| modelo.keras | Modelo selecionado com pré-processamento incorporado |
| extracao.weights.h5 | Melhor checkpoint da fase com base congelada |
| fine_tuning.weights.h5 | Melhor checkpoint da fase de fine-tuning |

Esses arquivos são gerados pelo código, não preenchidos com valores esperados. O .gitignore evita adicionar pesos e resultados acidentalmente ao git. Se quiser publicar gráficos/JSON de uma execução própria, selecione esses arquivos explicitamente. Os arquivos de uma execução registrada nesta entrega ficam em evidencias/, quando disponíveis, com seus limites descritos.
