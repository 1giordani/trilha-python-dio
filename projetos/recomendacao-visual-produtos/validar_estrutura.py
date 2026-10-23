"""Confere notebook e evidências sem repetir o treino nem baixar dados."""
import ast
import json
import re
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    for path in root.glob("*.py"): ast.parse(path.read_text())
    notebook=json.loads((root/"recomendacao_por_imagens.ipynb").read_text())
    assert notebook["nbformat"]==4
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            ast.parse(cell["source"])
            assert cell["execution_count"] is None and not cell["outputs"]
    metrics=json.loads((root/"evidencias/metricas.json").read_text())
    manifest=json.loads((root/"evidencias/manifesto.json").read_text())
    all_hashes=[h for group in manifest.values() for h in group["raw_hashes"]]
    assert len(all_hashes)==len(set(all_hashes))==10500
    for name,group in manifest.items():
        assert len(group["ids"])==metrics["counts"][name]
        assert len(group["raw_hashes"])==len(group["color_ids"])==len(group["ids"])
    rankings=json.loads((root/"evidencias/ranking-consultas.json").read_text())
    assert len(rankings["indices"])==500
    assert set(rankings["query_ids"]).isdisjoint(rankings["catalog_ids"])
    for ranks,scores in zip(rankings["indices"],rankings["similarities"]):
        assert len(ranks)==len(set(ranks))==len(scores)==5
        assert all(0<=index<1000 for index in ranks)
        assert all(-1.00001<=value<=1.00001 for value in scores)
        assert all(scores[i]>=scores[i+1]-1e-6 for i in range(4))
    assert metrics["ranking_uses_labels"] is False
    assert metrics["ranking_uses_text_or_price"] is False
    result={"status":"ok","notebook_cells":len(notebook["cells"]),"notebook_executed":False,
            "unique_original_images":len(all_hashes),"queries":500,"catalog_items":1000,
            "checks":["sintaxe", "hashes sem repetição exata entre grupos", "separação de IDs",
                      "cinco vizinhos distintos por consulta", "ordenação dos escores", "links locais"]}
    (root/"evidencias/estrutura.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for link in re.findall(r"\]\(([^)]+)\)",(root/"README.md").read_text()):
        if not link.startswith(("https://","http://","#")):
            assert (root/link).is_file(),f"Link quebrado: {link}"
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
