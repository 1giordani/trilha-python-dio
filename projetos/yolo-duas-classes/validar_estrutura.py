"""Conferência de arquivos, notebook e base empacotada; não repete treinamento."""
import ast
import json
import re
import zipfile
from pathlib import Path
from darknet_config import read_cfg


def main():
    root = Path(__file__).resolve().parent
    names = (root/"config/classes.names").read_text().splitlines()
    assert len(names) == 82
    assert names[80:] == ["marcador_triangulo", "marcador_quadrado"]
    blocks = read_cfg(root/"config/yolov3-tiny-82.cfg")
    heads = [i for i,b in enumerate(blocks) if b["type"] == "yolo"]
    assert len(heads) == 2
    assert all(blocks[i]["classes"] == "82" and blocks[i-1]["filters"] == "261" for i in heads)
    nb = json.loads((root/"yolo_duas_classes.ipynb").read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert cell["execution_count"] is None and not cell["outputs"]
    for path in root.glob("*.py"):
        ast.parse(path.read_text())
    with zipfile.ZipFile(root/"dataset-sintetico.zip") as archive:
        assert archive.testzip() is None
        images = [n for n in archive.namelist() if n.endswith(".png")]
        assert len(images) == 120
        for split, count in [("train",80),("valid",20),("test",20)]:
            assert len([n for n in images if f"images/{split}/" in n]) == count
        for path in archive.namelist():
            if "/labels/" in path and path.endswith(".txt"):
                rows = archive.read(path).decode().splitlines()
                assert {int(row.split()[0]) for row in rows} == {80,81}
    result = {"status":"ok", "classes":len(names), "yolo_heads":len(heads),
              "dataset_images":120, "notebook_cells":len(nb["cells"]),
              "notebook_executed":False,
              "checks":["sintaxe e esquema básico", "configuração 82/261", "IDs 80 e 81",
                        "ZIP íntegro", "separação 80/20/20", "links locais"]}
    (root/"evidencias/estrutura.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for link in re.findall(r"\]\(([^)]+)\)", (root/"README.md").read_text()):
        if not link.startswith(("https://","http://","#")):
            assert (root/link).is_file(), f"Link quebrado: {link}"
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
