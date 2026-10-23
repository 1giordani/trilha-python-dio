"""Valida e executa as células Python do notebook, sem serviços externos."""
import ast
import contextlib
import io
import json
import os
import re
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    nb = json.loads((root / "metricas_avaliacao.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
    source = (root / "metricas.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
    assert source in [c["source"] for c in cells], "Regere o notebook."
    namespace = {"__name__": "__main__"}
    with tempfile.TemporaryDirectory() as directory:
        previous = Path.cwd()
        try:
            os.chdir(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in cells:
                    ast.parse(cell["source"])
                    exec(compile(cell["source"], "notebook", "exec"), namespace)
            report = namespace["resultado"]
            assert report["total"] == 100
            assert report["metricas"]["acuracia"] == 0.85
            assert Path("resultados/metricas.json").is_file()
        finally:
            os.chdir(previous)
    result = {"status": "ok", "cells": len(nb["cells"]), "python_cells_executed": len(cells),
              "scope": "execução local das células; sem sessão no Colab",
              "checks": ["sintaxe", "sincronização com script", "execução integral das células",
                         "cálculos manuais", "arquivos gerados", "links locais do README"]}
    (root / "evidencias").mkdir(exist_ok=True)
    (root / "evidencias/validacao.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)", (root / "README.md").read_text(encoding="utf-8")):
        if not link.startswith(("https://", "http://", "#")):
            assert (root / link).is_file(), f"Link inválido: {link}"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
