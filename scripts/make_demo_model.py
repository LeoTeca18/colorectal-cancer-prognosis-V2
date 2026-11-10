"""Sincroniza os modelos oficiais do Notebook (`artifacts/`) para `models/`.

Mantido por compatibilidade com fluxos que invocavam `scripts/make_demo_model.py`.
Delega diretamente para `scripts.export_notebook_models.export_all()` de modo a
garantir 100% de uniformidade com as 43 variáveis codificadas do Notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_notebook_models import export_all

if __name__ == "__main__":
    export_all()
