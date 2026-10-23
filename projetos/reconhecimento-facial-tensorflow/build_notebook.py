"""Roteiro Colab para treino e inferência em várias faces."""
import json
from pathlib import Path

cells=[]


def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":source})


def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":source,"execution_count":None,"outputs":[]})


md("""# Sistema de detecção e reconhecimento facial — DIO
**Autor:** Giordani Martins Silva · implementação e documentação com auxílio de IA.

Usamos uma **MTCNN pré-treinada** para encontrar todas as faces e uma **CNN TensorFlow treinada do zero** para classificar oito identidades da base de pesquisa Olivetti. As classes são identificadores de laboratório, não nomes reais.

Execute as células na ordem. O piloto roda em CPU. Não é um sistema de autenticação, não tem vivacidade e pode aceitar pessoas não cadastradas. Para dados próprios, faça cadastro supervisionado de participantes autorizados; não publique suas fotos ou seus nomes automaticamente no GitHub.
""")
code('''import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","tensorflow==2.19.1","mtcnn==1.0.0",
                "numpy>=1.26,<2.2","scikit-learn>=1.5,<2","Pillow>=10.1,<13"],check=True)
''')
md("## Preparação\nSe o Colab solicitar reinício após a instalação, reinicie antes dos imports seguintes. Use Python 3.11/3.12. O código será baixado do seu repositório público; as fotos de pesquisa serão obtidas pelo loader oficial do scikit-learn.")
code('''from pathlib import Path
import os
repo=Path("/content/trilha-python-dio")
if not repo.exists():
    subprocess.run(["git","clone","https://github.com/1giordani/trilha-python-dio.git",str(repo)],check=True)
project=repo/"projetos/reconhecimento-facial-tensorflow"
os.chdir(project)
print(Path.cwd())
''')
md("## 1. Treinar e avaliar\nA separação por imagens ocorre antes da detecção. Identidades conhecidas 0–7 têm fotos distintas em treino/validação/teste. Identidades 8–9 ficam só na calibração de desconhecidos, e 10–11 só no teste de desconhecidos. O conjunto é pequeno e não representa uma avaliação em ambiente aberto.")
code('''subprocess.run([sys.executable,"train.py","--epochs","40"],check=True)
''')
md("## 2. Resultados\nConfira a acurácia de classificação fechada separadamente da aceitação correta com rejeição. Acurácia nos recortes detectados não mede a taxa de detecção de faces. Fotos nas quais a MTCNN falha são registradas em skipped, sem substituição silenciosa por um recorte conhecido.")
code('''import json
metrics=json.loads(Path("evidencias/metricas.json").read_text())
for name in ["used_counts","epochs_executed","known_closed_set_accuracy",
             "known_correct_acceptance","unknown_false_acceptance_rate","unknown_rejection_rate"]:
    print(name,metrics[name])
''')
md("## 3. Várias faces na mesma imagem\nA montagem usa amostras de validação e calibração para ilustração, não uma foto natural de grupo nem um teste independente. As caixas e rótulos são previsões reais do pipeline. Compare o JSON com os índices de origem para observar erros e rejeições.")
code('''from IPython.display import display
from PIL import Image
display(Image.open("evidencias/multiplas_faces.png"))
print(Path("evidencias/multiplas_faces.json").read_text())
''')
md("## 4. Outra imagem local (opcional)\nO modelo de demonstração conhece apenas as oito classes Olivetti. Uma foto sua não está cadastrada; o limiar é uma heurística e pode falhar. Use a etapa seguinte para treinar suas próprias classes.")
code('''image_path="" # Exemplo: /content/foto_do_laboratorio.jpg
if image_path:
    subprocess.run([sys.executable,"infer.py",image_path],check=True)
    display(Image.open("resultados/reconhecimento.png"))
''')
md("## 5. Cadastro próprio (opcional)\nOrganize dados-pessoais/pessoa_a/*.jpg e dados-pessoais/pessoa_b/*.jpg, com pelo menos dez fotos individuais por classe. Mantenha fotos de diferentes sessões e reserve imagens para validação e teste; o script separa arquivos, não agrupa sessões automaticamente. Ele exige exatamente uma face detectável por foto de cadastro. Não envie esses dados para o git.")
code('''treinar_cadastro_proprio=False
if treinar_cadastro_proprio:
    subprocess.run([sys.executable,"train_custom.py","--data","dados-pessoais","--epochs","40"],check=True)
    # Depois: infer.py foto.jpg --model artefatos-personalizados/classificador.keras
    #                            --metadata artefatos-personalizados/metadata.json
''')
md("## 6. Salvar modelo e evidências\nO arquivo inclui o classificador treinado e metadados necessários à inferência. Não inclui automaticamente as fotos privadas do cadastro próprio.")
code('''import zipfile
archive=project/"resultado-facial.zip"
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in ["artefatos","evidencias"]:
        for path in Path(folder).rglob("*"):
            if path.is_file(): z.write(path,str(path))
from google.colab import files
files.download(str(archive))
''')
md("""## Fontes e limites
- Faces de pesquisa: **AT&T Laboratories Cambridge / Olivetti**, via [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
- Detector: [ipazc/mtcnn](https://github.com/ipazc/mtcnn), redes PNet/RNet/ONet com pesos pré-treinados.
- Classificador: CNN em TensorFlow/Keras criada neste projeto, sem pesos de reconhecimento facial pré-treinados.

O sistema usa identificadores de cadastro e não implementa teste de vivacidade. O limiar softmax não garante reconhecer desconhecidos: veja falsos aceites registrados. O alinhamento usa a posição dos dois olhos para uma transformação de similaridade antes da classificação; se os landmarks não estiverem disponíveis, há recorte por caixa como alternativa.

Este notebook é entregue sem saídas de execução para reprodução limpa. O treinamento registrado na entrega foi local, não uma sessão no Colab. As aulas devem ser assistidas pelo estudante.
""")
for i,cell in enumerate(cells): cell["id"]=f"face-{i:02d}"
nb={"nbformat":4,"nbformat_minor":5,"cells":cells,"metadata":{
    "kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"},
    "colab":{"name":"reconhecimento_facial.ipynb"}}}
Path("reconhecimento_facial.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=2))
print("Notebook:",len(cells),"células")
