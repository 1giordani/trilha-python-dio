"""Validação estática e de consistência das evidências; não repete o treinamento."""
import ast
import json
import re
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    for path in root.glob("*.py"):
        ast.parse(path.read_text())
    notebook=json.loads((root/"reconhecimento_facial.ipynb").read_text())
    assert notebook["nbformat"]==4
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            ast.parse(cell["source"])
            assert cell["execution_count"] is None and not cell["outputs"]
    metrics=json.loads((root/"evidencias/metricas.json").read_text())
    groups=metrics["split_indices"]
    indices=[i for values in groups.values() for i in values]
    assert len(indices)==len(set(indices))==120
    for group,used in metrics["used_indices"].items():
        assert set(used).issubset(groups[group])
        assert len(used)==metrics["used_counts"][group]
    demo=json.loads((root/"evidencias/multiplas_faces.json").read_text())
    assert demo["detected_faces"]==len(demo["detections"])>=2
    correct_known=sum(d["matches_expected"] and d["label"]!="nao_cadastrado" for d in demo["detections"])
    assert correct_known>=2
    result={"status":"ok","notebook_cells":len(notebook["cells"]),"notebook_executed":False,
            "multiple_faces_detected":demo["detected_faces"],"correct_known_labels_in_demo":correct_known,
            "checks":["sintaxe", "grupos sem imagens repetidas", "contagens das evidências",
                      "mais de uma face reconhecida", "links locais"]}
    (root/"evidencias/estrutura.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for link in re.findall(r"\]\(([^)]+)\)",(root/"README.md").read_text()):
        if not link.startswith(("https://","http://","#")):
            assert (root/link).is_file(),f"Link inválido: {link}"
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
