"""Gera roteiro executável para reproduzir o piloto CPU no Colab."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type":"markdown", "metadata":{}, "source":text})


def code(text):
    cells.append({"cell_type":"code", "metadata":{}, "source":text, "execution_count":None, "outputs":[]})


md("""# Detecção YOLOv3-tiny — COCO80 + duas classes novas
**Autor:** Giordani Martins Silva · desafio DIO. Material preparado com auxílio de IA.

Este roteiro reproduz o **piloto sintético** executado localmente: 150 iterações, 128×128, batch 4, duas classes novas. Usa Darknet original, compilado em CPU para evitar dependência de uma versão antiga de CUDA. Não é necessário GPU. O tempo varia com a máquina.

As anotações de demonstração são geradas por código no formato XML do LabelMe e convertidas para YOLO. Não são uma sessão de rotulagem humana. Consulte docs/rotulagem.md para usar fotos e o software solicitado.
""")
code('''import subprocess
from pathlib import Path
import os

repo = Path("/content/trilha-python-dio")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/1giordani/trilha-python-dio.git", str(repo)], check=True)
project = repo / "projetos/yolo-duas-classes"
os.chdir(project)
subprocess.run(["python", "-m", "pip", "install", "-r", "requirements.txt"], check=True)
print(Path.cwd())
''')
md("## 1. Compilar o Darknet\nO commit é fixado para manter compatibilidade do adaptador ctypes. Não troque por um fork diferente mantendo a mesma estrutura de memória. Este notebook compila com GPU=0 e OpenCV=0.")
code('''darknet = project / "darknet"
commit = "f6afaabcdf85f77e7aff2ec55c020c0e297c77f9"
if not darknet.exists():
    subprocess.run(["git", "clone", "https://github.com/pjreddie/darknet.git", str(darknet)], check=True)
subprocess.run(["git", "-C", str(darknet), "checkout", commit], check=True)
subprocess.run(["make", "-j2", "GPU=0", "CUDNN=0", "OPENCV=0"], cwd=darknet, check=True)
''')
md("## 2. Criar a base e configurar 82 classes\nOs IDs COCO 0–79 continuam na lista. As novas classes são 80=marcador_triangulo e 81=marcador_quadrado. A base tem 80/20/20 imagens de treino/validação/teste; cada imagem contém os dois objetos.")
code('''subprocess.run(["python", "prepare.py", "--darknet", str(darknet),
                "--iterations", "150", "--size", "128", "--batch", "4"], check=True)
''')
md("## 3. Transferir os pesos\nCada cabeça usa 3×(5+82)=261 filtros. O transplante copia por âncora os 5 valores de caixa/objectness e os 80 valores de classes anteriores. As duas saídas novas são inicializadas separadamente. O contador de imagens é zerado para iniciar o novo treino. Download: fonte original primeiro, mirror documentado com hash como alternativa.")
code('''subprocess.run(["python", "download_weights.py"], check=True)
subprocess.run(["python", "expand_weights.py", str(darknet / "cfg/yolov3-tiny.cfg"),
                "config/yolov3-tiny-82.cfg", "weights/yolov3-tiny.weights",
                "weights/yolov3-tiny-82-init.weights"], check=True)
''')
md("## 4. Treinar de verdade\nO log será salvo para revisão. O número de iterações é definido antes de consultar validação/teste. A execução pode demorar alguns minutos. Não interrompa antes de gerar o arquivo final se quiser seguir as próximas células.")
code('''Path("evidencias").mkdir(exist_ok=True)
with open("evidencias/treino-colab.log", "w") as log:
    subprocess.run([str(darknet / "darknet"), "detector", "train", "config/dataset.data",
                    "config/yolov3-tiny-82.cfg", "weights/yolov3-tiny-82-init.weights"],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print("\\n".join(Path("evidencias/treino-colab.log").read_text().splitlines()[-8:]))
''')
md("## 5. Avaliar e visualizar\nAP50 é calculada somente nas duas classes novas. Não é COCO mAP@[.5:.95] e não demonstra preservação das 80 classes antigas após o retreino. Não ajuste hiperparâmetros pelo teste.")
code('''for split in ["valid", "test"]:
    subprocess.run(["python", "evaluate.py", "--darknet", str(darknet),
                    "--weights", "backup/yolov3-tiny-82_final.weights", "--split", split,
                    "--output", f"evidencias/{split}-colab"], check=True)
from IPython.display import display
from PIL import Image
display(Image.open("evidencias/test-colab/predicao.png"))
print(Path("evidencias/test-colab/metricas.json").read_text())
''')
md("## 6. Salvar seu trabalho\nO ZIP inclui os pesos finais e dados que serão perdidos ao encerrar a sessão. Os pesos não são adicionados ao git automaticamente.")
code('''import zipfile
archive = project / "resultado-yolo.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
    for folder in ["backup", "evidencias", "config"]:
        for path in Path(folder).rglob("*"):
            if path.is_file():
                zipped.write(path, str(path))
from google.colab import files
files.download(str(archive))
''')
md("""## Evoluir para fotos reais
1. Rotule no LabelMe, usando exatamente os nomes das duas classes novas.
2. Converta XML/JSON com annotations.py e separe por origem/cena antes de treinar.
3. Atualize os caminhos train/valid/test e revise todas as caixas.
4. Use validação para escolher um orçamento de treino apropriado; o piloto de 150 iterações não é uma recomendação universal.
5. Para avaliar retenção das classes antigas, adicione imagens COCO com rótulos completos ao treino (replay) e conjuntos de avaliação independentes. Não declare as 80 classes preservadas apenas porque seus nomes e pesos iniciais foram copiados.

O exemplo sintético tem viés espacial (triângulos à esquerda e quadrados à direita) e domínio simples. Bom resultado nele não comprova generalização para fotografias. A execução do notebook no Colab não foi realizada na preparação; o piloto registrado foi executado localmente em CPU.
""")
for i, cell in enumerate(cells):
    cell["id"] = f"yolo-{i:02d}"
nb = {"nbformat":4,"nbformat_minor":5,"cells":cells,
      "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
                  "language_info":{"name":"python"},"colab":{"name":"yolo_duas_classes.ipynb"}}}
Path("yolo_duas_classes.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=2))
print("Notebook:",len(cells),"células")
