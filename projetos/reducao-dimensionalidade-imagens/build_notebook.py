"""Cria um notebook autossuficiente, com o mesmo algoritmo do script."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source,
                  "execution_count": None, "outputs": []})


md("""# Redução de dimensionalidade em imagens — DIO
**Autor:** Giordani Martins Silva. Código e documentação preparados com auxílio de IA.

Objetivo: transformar RGB em um canal de cinza (0 a 255) e aplicar um limiar para obter preto e branco (0 ou 255). A transformação dos pixels é implementada explicitamente, sem chamar convert('L') nem uma função pronta de threshold.

Execute as células na ordem. Não precisa de GPU. O exemplo padrão é gerado localmente e não depende de download de fotos. É possível enviar sua própria imagem na etapa 3.
""")
code("%pip -q install 'Pillow>=10.1,<13'")
md("## 1. Algoritmo\nCinza = arredondamento de 0,299R + 0,587G + 0,114B. A implementação usa inteiros e soma 500 antes de dividir por 1000 para arredondar. Em seguida: cinza >= limiar vira 255; abaixo vira 0. O padrão é 128.")
source = (root / "processar.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
code(source)
md("## 2. Gerar uma imagem colorida de exemplo\nO gradiente e as faixas de cores permitem observar que cores diferentes podem virar o mesmo nível de cinza ou a mesma classe binária.")
sample = (root / "gerar_exemplo.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
code(sample + '\nentrada = criar_exemplo()\n')
md("## 3. Usar imagem própria (opcional)\nMude usar_imagem_propria para True para abrir a seleção de arquivo no Colab. PNG e JPEG são aceitos. Imagens com transparência são compostas sobre branco; a orientação EXIF é respeitada.")
code('''usar_imagem_propria = False
if usar_imagem_propria:
    from google.colab import files
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("Nenhum arquivo foi enviado.")
    entrada = Path(next(iter(uploaded)))
print("Entrada:", entrada)
''')
md("## 4. Processar e conferir\nExperimente 64, 128 e 192. Quanto maior o limiar, menor ou igual é a quantidade de pixels brancos. A igualdade com o limiar é classificada como branco.")
code('''limiar = 128
relatorio = processar(entrada, "resultados", limiar)
print(json.dumps(relatorio, ensure_ascii=False, indent=2))
assert set(relatorio["valores_binaria"]).issubset({0, 255})
assert relatorio["dimensoes_preservadas"]
assert relatorio["pixels_pretos"] + relatorio["pixels_brancos"] == relatorio["pixels"]
''')
md("## 5. Visualizar os três resultados")
code('''try:
    from IPython.display import display
except ImportError:
    print("Comparação salva em resultados/comparacao.png")
else:
    with Image.open("resultados/comparacao.png") as preview:
        display(preview.copy())
''')
md("## 6. Baixar os arquivos\nO ZIP contém original normalizada, cinza, binária, comparação e relatório JSON.")
code('''import shutil
archive = shutil.make_archive("imagens-processadas", "zip", "resultados")
try:
    from google.colab import files
except ImportError:
    print("Arquivo gerado:", archive)
else:
    files.download(archive)
''')
md("""## O que foi reduzido?
A imagem RGB passa de três canais para um. A binarização reduz os valores possíveis para dois. **Não há redução de largura ou altura**, nem aplicação de PCA. As saídas PNG são salvas em modo L de 8 bits: ter dois valores não significa que o arquivo foi salvo com um bit por pixel.

A fórmula é uma aproximação usual de luma sobre valores RGB, sem linearização de gamma; não equivale a medir luminância física. Um limiar global pode perder detalhes em imagens com iluminação desigual.

O notebook é distribuído sem saídas para nova execução. As evidências executadas localmente estão no repositório. Este material não comprova visualização das aulas nem uma sessão no Colab.
""")
for i, cell in enumerate(cells):
    cell["id"] = f"img-{i:02d}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "colab": {"name": "reducao_dimensionalidade_imagens.ipynb"}}}
(root / "reducao_dimensionalidade_imagens.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook gerado: {len(cells)} células")
