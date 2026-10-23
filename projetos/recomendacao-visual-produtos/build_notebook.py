"""Cria notebook de treinamento e recomendação visual para Colab."""
import json
from pathlib import Path

cells=[]


def md(source): cells.append({"cell_type":"markdown","metadata":{},"source":source})


def code(source): cells.append({"cell_type":"code","metadata":{},"source":source,"execution_count":None,"outputs":[]})


md("""# Sistema de recomendação por imagens — DIO
**Autor:** Giordani Martins Silva · implementação e documentação com auxílio de IA.

Treinamos uma CNN em dez categorias de moda e usamos seu embedding como representação de formato/aparência. Um histograma RGB acrescenta informação explícita de cor. O ranking usa similaridade cosseno, sem preço, marca, descrição textual ou filtro por rótulo.

Base: Fashion-MNIST / Zalando Research. Ela é originalmente cinza; as cores deste laboratório são **artificiais**, atribuídas independentemente da classe para demonstrar o componente cromático. Não são cores reais de produtos.
""")
code('''import subprocess,sys
subprocess.run([sys.executable,"-m","pip","install","tensorflow==2.19.1","numpy>=1.26,<2.2","Pillow>=10.1,<13"],check=True)
''')
md("## Preparar\nUse Python 3.11/3.12. Se o Colab pedir reinício após instalar, reinicie antes dos imports. O piloto pode rodar em CPU; tempo depende da máquina.")
code('''from pathlib import Path
import os
repo=Path("/content/trilha-python-dio")
if not repo.exists():
    subprocess.run(["git","clone","https://github.com/1giordani/trilha-python-dio.git",str(repo)],check=True)
os.chdir(repo/"projetos/recomendacao-visual-produtos")
print(Path.cwd())
''')
md("## 1. Treinar a rede multiclasse\nSão 8.000 exemplos para treino e 1.000 para validação; 1.000 itens formam o catálogo e 500 ficam como consultas. Imagens originais idênticas são removidas entre todos os grupos, antes da colorização. A melhor época é escolhida por validação, sem usar as consultas de teste.")
code('''subprocess.run([sys.executable,"train.py","--epochs","5"],check=True)
''')
md("## 2. Avaliar o ranking\nPrecision@5 mede a proporção de itens da mesma categoria entre os cinco primeiros. Isso é um proxy de relevância, não uma avaliação humana de formato, estilo, textura ou intenção de compra. Os rótulos são usados aqui para avaliar, não para ordenar.")
code('''import json
metrics=json.loads(Path("evidencias/metricas.json").read_text())
for key in ["counts","classifier_accuracy_on_queries","precision_at_5_class_proxy",
            "hit_rate_at_5_class_proxy","random_precision_at_5_class_proxy"]:
    print(key,metrics[key])
from IPython.display import display
from PIL import Image
display(Image.open("evidencias/recomendacoes.png"))
''')
md("## 3. Consultar uma imagem\nEnvie uma imagem própria ou salva de uma fonte que permita seu uso. O modelo está restrito ao domínio simples de moda; uma foto arbitrária de internet pode ter fundo, escala e objetos não vistos no treino. O sistema consulta o catálogo local, sem buscar lojas na internet.")
code('''import numpy as np
from recommend import load_system,recommend
from figures import recommendation_grid
encoder,catalog=load_system()
enviar_imagem=False
if enviar_imagem:
    from google.colab import files
    uploaded=files.upload()
    if not uploaded: raise ValueError("Nenhuma imagem enviada")
    query_image=Image.open(next(iter(uploaded)))
else:
    queries=np.load("artefatos/consultas.npz",allow_pickle=False)
    query_image=Image.fromarray(queries["images"][0])
query,indices,scores,items=recommend(encoder,catalog,query_image,k=5)
Path("resultados").mkdir(exist_ok=True)
recommendation_grid([query],catalog["images"],[indices],[scores],"resultados/minha_consulta.png")
display(Image.open("resultados/minha_consulta.png"))
print(json.dumps(items,indent=2))
''')
md("## 4. Indexar catálogo próprio (opcional)\nColoque as fotos em catalogo-proprio/. Nomes de arquivos são somente identificadores. O script extrai os mesmos descritores das imagens, sem ler preços ou marca. A CNN continua sendo a do laboratório: para catálogos reais, treine/valide representações apropriadas e revise o pré-processamento.")
code('''indexar_catalogo_proprio=False
if indexar_catalogo_proprio:
    subprocess.run([sys.executable,"index_catalog.py","catalogo-proprio"],check=True)
    # Consultar depois com recommend.py imagem.jpg --catalog artefatos/catalogo-proprio.npz
''')
md("## 5. Baixar modelo e índice\nO catálogo indexado e encoder.keras são suficientes para busca. A exportação também inclui o classificador e evidências. O arquivo pode conter imagens do laboratório; não publique fotos privadas de um catálogo próprio sem revisão.")
code('''import zipfile
archive=Path("recomendador-treinado.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in ["artefatos","evidencias","resultados"]:
        for path in Path(folder).rglob("*"):
            if path.is_file(): z.write(path,str(path))
from google.colab import files
files.download(str(archive))
''')
md("""## Interface para usuário de um site
O repositório inclui app.py em Streamlit. Para rodar no computador:

```bash
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

Ela permite escolher um item ou enviar uma imagem e ver cinco recomendações. Não há hospedagem pública automática nesta entrega. No Colab, a consulta visual acima já demonstra a função sem abrir portas ou usar túneis.

## Interpretação e fontes
O descritor final tem 88 dimensões: 64 do embedding e 24 do histograma (8 bins por canal). Após normalização, a concatenação usa pesos de energia 75%/25%; a similaridade é a combinação ponderada dos cossenos dos dois componentes.

Não confunda similaridade com probabilidade de compra. A resolução 28×28 e cores artificiais limitam textura e validade fora do domínio. A aprendizagem é supervisionada por categoria, não por pares de preferência visual.

Fontes: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data),
[notebook indicado no enunciado](https://colab.research.google.com/github/sparsh-ai/rec-tutorials/blob/master/_notebooks/2021-04-27-image-similarity-recommendations.ipynb).

Os outputs são deixados vazios para execução limpa no Colab. As evidências publicadas vieram de execução local. Aulas devem ser assistidas pelo estudante.
""")
for i,c in enumerate(cells): c["id"]=f"rec-{i:02d}"
nb={"nbformat":4,"nbformat_minor":5,"cells":cells,"metadata":{
    "kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"},
    "colab":{"name":"recomendacao_por_imagens.ipynb"}}}
Path("recomendacao_por_imagens.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=2))
print("Notebook:",len(cells),"células")
