"""Tests de la Fase 5 (tablas del informe).

El riesgo real de esta fase no es que un número salga mal --- viene de los
CSV de Fase 3, que ya están probados --- sino que el LaTeX generado no
compile o imprima marcado en crudo. Eso pasa por dos motivos, y hay un test
para cada uno:

    1. Un carácter especial sin escapar (`_`, `%`, `>`) revienta pdflatex o
       lo mete en modo matemático.
    2. Un comando LaTeX metido dentro de una celda de datos sale impreso
       literalmente, porque el contenido se escapa (es dato, no marcado).

`run_fase5.py` importa matplotlib, que no todos los entornos de CI tienen;
si falta, estos tests se saltan en vez de fallar.
"""

import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="run_fase5 necesita matplotlib")

from run_fase5 import TABLAS_DIR, escapar_latex, tabla_latex  # noqa: E402


@pytest.mark.parametrize("crudo,esperado", [
    ("t_min_medio", r"t\_min\_medio"),
    (">120", r"\textgreater{}120"),
    ("50%", r"50\%"),
    ("A & B", r"A \& B"),
    ("100$", r"100\$"),
    ("#1", r"\#1"),
])
def test_escapar_latex(crudo, esperado):
    assert escapar_latex(crudo) == esperado


def test_escapar_latex_nulo_es_raya():
    assert escapar_latex(None) == "--"
    assert escapar_latex(float("nan")) == "--"


def test_tabla_latex_estructura_booktabs(tmp_path, monkeypatch):
    monkeypatch.setattr("run_fase5.TABLAS_DIR", tmp_path)
    df = pd.DataFrame({"banda": [">120"], "pct": [10.58]})
    dest = tabla_latex(df, ["banda", "pct"], ["Banda", "% de población"], "lr",
                       {"pct": lambda v: f"{v:.1f}"}, nombre="prueba")
    tex = dest.read_text(encoding="utf-8")

    assert tex.startswith(r"\begin{tabular}{lr}")
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 1
    assert r"\textgreater{}120 & 10.6 \\" in tex
    assert r"\% de población" in tex  # el encabezado también se escapa


def test_tabla_latex_escapa_el_contenido_no_lo_interpreta(tmp_path, monkeypatch):
    """Una celda con pinta de comando LaTeX debe salir escapada, no ejecutada:
    el contenido de una tabla es dato, y un dato no puede inyectar marcado."""
    monkeypatch.setattr("run_fase5.TABLAS_DIR", tmp_path)
    df = pd.DataFrame({"x": [r"\newpage"]})
    tex = tabla_latex(df, ["x"], ["X"], "l", nombre="prueba").read_text(encoding="utf-8")
    assert r"\textbackslash{}newpage" in tex
    assert r"\newpage" not in tex


@pytest.mark.artefactos
def test_tablas_generadas_no_tienen_marcado_impreso():
    """Regresión: los encabezados y las celdas se escapan, así que colar un
    `\\#` o un `$\\times$` en la definición de una tabla lo imprime literal.
    Ninguna tabla del informe debe contener un `\\textbackslash`."""
    generadas = sorted(TABLAS_DIR.glob("informe_*.tex"))
    if not generadas:
        pytest.skip("Faltan las tablas del informe; corre `python run_fase5.py`.")

    for tex_path in generadas:
        contenido = tex_path.read_text(encoding="utf-8")
        assert r"\textbackslash" not in contenido, (
            f"{tex_path.name} imprime una barra invertida: alguien metió un comando "
            "LaTeX en un encabezado o en una celda, que se escapan por diseño."
        )
        assert contenido.count(r"\begin{tabular}") == contenido.count(r"\end{tabular}") == 1
