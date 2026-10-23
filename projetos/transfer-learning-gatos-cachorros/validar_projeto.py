"""Validação estática: não executa treinamento nem baixa dados."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import nbformat
    path = ROOT / "transfer_learning_gatos_cachorros.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    for i, source in enumerate(code_cells):
        # %pip é uma instrução de IPython, e não Python puro.
        python_source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        ast.parse(python_source, filename=f"cell_{i}")
    sections = (ROOT / "pipeline.py").read_text(encoding="utf-8").split("# %%\n")[1:]
    for section in sections:
        assert section.strip() in code_cells, "Notebook desatualizado: execute build_notebook.py"
    for filename in ["pipeline.py", "train.py", "build_notebook.py", "test_model.py"]:
        ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    for link in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
        if not link.startswith(("https://", "http://", "#")):
            assert (ROOT / link).is_file(), f"Link local inválido: {link}"
    assert all(c.execution_count is None and not c.outputs for c in notebook.cells if c.cell_type == "code")
    print(json.dumps({"status": "ok", "cells": len(notebook.cells),
                      "checks": ["schema nbformat", "sintaxe de todas as células",
                                 "notebook sincronizado ao pipeline", "links locais do README",
                                 "saídas de treinamento não fabricadas"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
