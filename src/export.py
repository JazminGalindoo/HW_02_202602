"""
export.py — Fase 3: exportación de outputs de métricas
==========================================================

Responsabilidad separada de metrics.py a propósito (ver Tarea 4 de la
consigna): las funciones de metrics.py son puras (DataFrame(s) in ->
DataFrame(s) out, nunca tocan disco), así son testeables sin fixtures de
archivos y reusables desde el dashboard de Fase 4 sin re-ejecutar I/O. Este
módulo es la única capa que escribe:

    - data/outputs/*.csv        todos los outputs de métricas (para el
                                  dashboard de Streamlit, Fase 4)
    - report/tables/*.tex       el subconjunto de tablas citadas en el
                                  informe, con df.to_latex(booktabs=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config_loader import REPO_ROOT, load_config

logger = logging.getLogger("export")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Ver la nota equivalente en population.py/metrics.py sobre por qué no
    # usamos logging.basicConfig() aquí.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] export: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def export_csv(df: pd.DataFrame, name: str, cfg: Optional[dict] = None) -> Path:
    """Escribe df en data/outputs/{name}.csv (ruta base en config.md > rutas.outputs)."""
    cfg = cfg or load_config()
    dest = REPO_ROOT / cfg["rutas"]["outputs"] / f"{name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("CSV exportado: %s (%d filas, %d columnas)", dest, *df.shape)
    return dest


def export_latex(df: pd.DataFrame, name: str, cfg: Optional[dict] = None, **to_latex_kwargs) -> Path:
    """Escribe df en report/tables/{name}.tex (ruta base en
    config.md > rutas.report_tablas_dir) con reglas estilo booktabs
    (\\toprule/\\midrule/\\bottomrule).

    pandas >= 2.0 quitó el parámetro booktabs=True de DataFrame.to_latex()
    (ahora se controla vía la opción global 'styler.latex.hrules'); se activa
    aquí con pd.option_context para no dejar el estado global de pandas
    alterado fuera de esta función.

    Por defecto usa float_format="%.2f" y na_rep="--" para que las tablas
    del informe no salgan con 6+ decimales o 'NaN' crudo; ambos son
    sobreescribibles vía to_latex_kwargs si una tabla puntual lo necesita."""
    cfg = cfg or load_config()
    dest = REPO_ROOT / cfg["rutas"]["report_tablas_dir"] / f"{name}.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"index": False, "na_rep": "--", "float_format": "%.2f"}
    kwargs.update(to_latex_kwargs)
    with pd.option_context("styler.latex.hrules", True):
        latex = df.to_latex(**kwargs)
    dest.write_text(latex, encoding="utf-8")

    logger.info("Tabla LaTeX exportada: %s (%d filas, %d columnas)", dest, *df.shape)
    return dest
