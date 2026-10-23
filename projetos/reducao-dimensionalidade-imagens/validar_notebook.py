"""Executa as células Python padrão em diretório temporário, sem abrir o Colab."""
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
    nb = json.loads((root / "reducao_dimensionalidade_imagens.ipynb").read_text())
    assert nb["nbformat"] == 4 and len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    source = (root / "processar.py").read_text().split('\nif __name__ == "__main__":')[0]
    assert source in code, "Notebook precisa ser regenerado."
    namespace = {"__name__": "notebook"}
    with tempfile.TemporaryDirectory() as directory:
        previous = Path.cwd()
        try:
            os.chdir(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in code:
                    python = "\n".join(line for line in cell.splitlines() if not line.startswith("%"))
                    ast.parse(python)
                    exec(compile(python, "notebook", "exec"), namespace)
            report = namespace["relatorio"]
            assert report["valores_binaria"] == [0, 255]
            assert report["dimensoes_preservadas"]
            assert report["pixels"] == 98304
        finally:
            os.chdir(previous)
    output = {"status": "ok", "cells": len(nb["cells"]),
              "python_cells_executed": len(code),
              "scope": "Python local; sem sessão Colab; %pip omitido porque Pillow já instalado",
              "checks": ["sintaxe", "execução padrão do notebook", "sincronização com processar.py",
                         "somente 0 e 255", "dimensões preservadas", "links do README"]}
    (root / "evidencias/validacao.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    for link in re.findall(r"\]\(([^)]+)\)", (root / "README.md").read_text()):
        if not link.startswith(("https://", "http://", "#")):
            assert (root / link).is_file(), f"Link quebrado: {link}"
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
