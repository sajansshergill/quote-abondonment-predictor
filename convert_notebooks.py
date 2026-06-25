import os
import json
from uuid import uuid4

ROOT = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")

PY_NOTEBOOKS = [
    "01_eda.py",
    "02_survival_analysis.py",
    "03_abandonment_model.py",
    "04_intervention_sim.py",
]

def new_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def new_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }

def percent_script_to_notebook(source: str) -> dict:
    """Convert a lightweight # %% script into a notebook without jupytext."""
    cells = []
    current_lines = []
    current_type = "code"

    def flush_cell() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        source_text = "".join(current_lines).rstrip()
        if current_type == "markdown":
            source_text = "\n".join(
                line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                for line in source_text.splitlines()
            )
            cells.append(new_markdown_cell(source_text))
        else:
            cells.append(new_code_cell(source_text))
        current_lines = []

    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush_cell()
            current_type = "markdown" if "[markdown]" in stripped else "code"
            continue
        current_lines.append(line)

    flush_cell()
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def convert(filename: str):
    src = os.path.join(NOTEBOOKS_DIR, filename)
    out = src.replace(".py", ".ipynb")
    print(f"  Converting {filename} -> {os.path.basename(out)}")
    with open(src, encoding="utf-8") as f:
        notebook = percent_script_to_notebook(f.read())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        f.write("\n")
    print(f"  Done: {out}")

def main():
    print("\n-- Converting notebooks (.py -> .ipynb) ----------------------\n")
    for nb in PY_NOTEBOOKS:
        convert(nb)
    print("\nAll notebooks converted. Open the .ipynb files in JupyterLab or VS Code.\n")

if __name__ == "__main__":
    main()