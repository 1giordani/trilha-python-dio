# Evidências de execução

## Validações do código

O teste de integração em test_model.py foi executado com TensorFlow CPU 2.19.1, Python 3.12 e Keras 3.15.1. Resultado: **1 teste passou**. Ele verifica treino da cabeça, liberação de camadas com BatchNormalization congelada, recompilação, um passo de fine-tuning e equivalência de previsões após salvar e recarregar o modelo.

Esse teste usa **imagens sintéticas e pesos aleatórios** para validar a integração sem dependência de rede. Ele não é o experimento de Transfer Learning e não mede classificação de animais.

O arquivo validacao-estatica.json registra a validação do formato do notebook, sintaxe de suas células, sincronização com pipeline.py e vínculos locais da documentação.

## Experimento com dados reais

Foi executado um ensaio local com pesos ImageNet e a base real cats_vs_dogs:4.0.1, em CPU, no dia 23/10/2026. Configuração: imagens 96×96, treino limitado a 256, validação limitada a 64, teste limitado a 64, uma época de extração e uma de fine-tuning. O modelo de fine-tuning foi selecionado pela menor perda de validação.

Comando executado, a partir da pasta do projeto:

```bash
python train.py --head-epochs 1 --fine-epochs 1 --train-limit 256 --evaluation-limit 64 --image-size 96 --output-dir /tmp/tl-dio-resultados
```

Na máquina usada, também foram definidos TF_NUM_INTEROP_THREADS=2, TF_NUM_INTRAOP_THREADS=2 e MPLBACKEND=Agg para limitar paralelismo e gerar figuras sem janela. Isso não altera o protocolo de separação dos dados. O JSON preserva o caminho de saída original da execução.

Arquivos preservados em [ensaio-reduzido/](ensaio-reduzido/): metricas.json, historico.json, predicoes.csv, classes.json, curvas.png e matriz_confusao.png. O gráfico das curvas foi redesenhado a partir do histórico real para exibir marcadores: com uma única época por fase, linhas sem marcadores não mostravam os pontos. Não foram alterados os dados. A acurácia observada foi 57/64 (89,0625%); o teste tinha 48 gatos e 16 cachorros. A matriz contém [[44,4],[3,13]], com linhas verdadeiras e colunas previstas na ordem cat/dog. O modelo salvo também foi recarregado e executou inferência com saída finita.

O notebook padrão tem configuração de 5+3 épocas e conjuntos completos; seus outputs são deixados vazios para execução limpa no Colab. **O experimento completo não foi executado** nesta preparação. As métricas do ensaio reduzido não substituem suas métricas. Os pesos e fotos da base não são adicionados ao repositório.

Não houve sessão no Colab nem prints de aulas nesta preparação. O teste local e qualquer ensaio registrado são evidências distintas da execução no Colab e do experimento completo.
