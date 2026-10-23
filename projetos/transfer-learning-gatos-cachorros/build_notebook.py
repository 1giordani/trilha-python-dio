"""Gera notebook autossuficiente a partir das funções de pipeline.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text,
                  "execution_count": None, "outputs": []})


markdown('''# Transfer Learning — gatos e cachorros
**Autor:** Giordani Martins Silva · desafio de projeto DIO

Objetivo: adaptar uma MobileNetV2 pré-treinada no ImageNet para classificação binária.
O projeto diferencia **extração de características** de **fine-tuning** e mantém
o teste separado da escolha do modelo. Código e documentação preparados com auxílio de IA.

**Execução:** no Colab, escolha um ambiente com GPU se disponível e execute todas as células em ordem.
O download inicial da base é de aproximadamente 787 MiB, além dos pesos. O tempo varia conforme hardware.
Este notebook é entregue sem saídas de treinamento: métricas só serão exibidas após execução real.
O link histórico oferecido pela DIO é uma referência; esta implementação usa TensorFlow/Keras e gatos/cachorros.
''')
code('''# Ambiente de referência testado: Python 3.12 e TensorFlow 2.19.1.
# Se o Colab pedir reinício após a instalação, reinicie antes dos imports.
%pip -q install tensorflow==2.19.1 tensorflow-datasets==4.9.9 tensorflow-metadata==1.16.1 importlib-resources==7.1.0 "matplotlib>=3.8,<4" "scikit-learn>=1.5,<2"
''')
sections = (ROOT / "pipeline.py").read_text(encoding="utf-8").split("# %%\n")
titles = [
    "## 1. Configuração e reprodutibilidade\nSemente fixa e registro de versões ajudam a comparar execuções; GPU e bibliotecas ainda podem produzir variações.",
    "## 2. Dados e separação\nUsamos faixas disjuntas 80/10/10 do split original antes do embaralhamento. Não há cache de todas as imagens em memória. O TFDS já trata imagens corrompidas dessa versão. A divisão não é estratificada e não garante ausência de imagens parecidas entre conjuntos.",
    "## 3. Modelo\nA MobileNetV2 recebe imagens em [-1, 1]. Aumento de dados funciona somente no treino. A saída sigmoid representa a classe de índice 1 lida dos metadados. BatchNormalization permanece em inferência durante o fine-tuning.",
    "## 4. Duas etapas e seleção\nTreinamos a cabeça, depois liberamos as últimas 30 camadas exceto BatchNormalization e recompilamos com taxa menor. Guardamos a melhor época de cada fase e escolhemos por menor perda de validação. Fine-tuning não tem melhoria garantida.",
    "## 5. Avaliação, exportação e inferência\nAcurácia e AUC são complementadas por precisão, recall, F1 e matriz de confusão. Não ajuste hiperparâmetros usando o teste. O escore sigmoid não é garantia de confiança calibrada.",
]
for title, section in zip(titles, sections[1:]):
    markdown(title)
    code(section.strip())
markdown("## 6. Carregar dados e iniciar\nO valor train_limit=0 usa todo o treino. Para um ensaio rápido, use 512; esse limite será registrado e o resultado não equivale ao experimento completo.")
code('''config = Config()
setup(config)
(train, validation, test), metadata = load_data(config)
show_samples(train, metadata["classes"], config.output_dir)
model, base = build_model(config)  # weights="imagenet": transferência real
model.summary()
''')
markdown("## 7. Treinar\nAs perdas e acurácias impressas abaixo serão os resultados da sua execução, sem números predefinidos.")
code('''histories, comparison = train_stages(model, base, train, validation, config)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
plot_history(histories, config.output_dir)
''')
markdown("## 8. Avaliar o modelo selecionado\nO teste é usado pela primeira vez nesta etapa. O pipeline salva o modelo, mapeamento de classes, métricas, predições e gráficos.")
code('''results = evaluate_and_export(model, test, metadata, histories, comparison, config)
''')
markdown("## 9. Conferir o modelo recarregado")
code('''reloaded = tf.keras.models.load_model(Path(config.output_dir) / "modelo.keras")
images, _ = next(iter(test))
np.testing.assert_allclose(model(images, training=False).numpy(),
                           reloaded(images, training=False).numpy(), atol=1e-5)
print("Modelo exportado e recarregado: previsões equivalentes.")
''')
markdown("## 10. Imagem própria (opcional)\nEnvie uma foto de gato ou cachorro pelo painel de arquivos. Defina seu caminho abaixo. O modelo sempre escolhe uma das duas classes, inclusive para imagens fora do escopo.")
code('''image_path = ""  # Exemplo: /content/meu_gato.jpg
if image_path:
    print(classify_image(reloaded, image_path, metadata["classes"], config.image_size))
''')
markdown("## 11. Baixar as evidências\nO ZIP preserva métricas e arquivos que somem ao encerrar a sessão do Colab. Para GitHub, publique gráficos e JSON; guarde arquivos grandes do modelo fora do git ou em uma release.")
code('''import shutil
archive = shutil.make_archive("evidencias-transfer-learning", "zip", config.output_dir)
try:
    from google.colab import files
except ImportError:
    print("Arquivo disponível em:", archive)
else:
    files.download(archive)
''')
markdown('''## Reflexão e fontes
- Compare perda de validação das etapas: o fine-tuning ajudou nesta execução?
- Observe a classe com menor recall e imagens com erro. Fundo, iluminação e domínio podem influenciar.
- Registre hardware, versões, tempo observado e resultados sem confundir validação com teste.
- As aulas devem ser assistidas pelo estudante; este material não comprova essa etapa.

Referências: [TensorFlow Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning),
[Cats vs Dogs/TFDS](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs),
[MobileNetV2](https://keras.io/api/applications/mobilenet/),
[Microsoft Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765),
[notebook de referência da DIO](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb).
''')
for i, cell in enumerate(cells):
    cell["id"] = f"tl-{i:03d}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"colab": {"name": "transfer_learning_gatos_cachorros.ipynb"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"}}
(ROOT / "transfer_learning_gatos_cachorros.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook gerado: {len(cells)} células")
