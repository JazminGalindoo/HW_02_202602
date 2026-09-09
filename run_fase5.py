"""
run_fase5.py — Fase 5: figuras y tablas del informe
=====================================================

Genera, a partir de los CSV de `data/outputs/` y de los artefactos de Fase 4
(única fuente de verdad: no recalcula nada por su cuenta salvo agregados
descriptivos de presentación):

    report/figures/*.pdf (+.png) las 8 figuras que cita `report/informe.tex`
    report/tables/informe_*.tex tablas booktabs listas para \\input

Por qué tablas nuevas y no las de Fase 3: `src/export.py` vuelca los
DataFrames tal cual, con los nombres de columna del pipeline
(`t_min_medio_ponderado`, `pct_poblacion`...). Esos guiones bajos son
caracteres especiales en LaTeX y no compilan dentro de un documento; además
un informe no se lee con nombres de variable por encabezado. Las tablas de
`report/tables/*.tex` de Fase 3 se conservan intactas como volcado
reproducible; las `informe_*.tex` que produce este script son su versión
presentable, escapada y en castellano. Ambas salen del mismo CSV, así que no
pueden divergir.

Uso:
    python run_fase5.py
    # y luego, para el PDF:
    cd report && pdflatex informe.tex && pdflatex informe.tex
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # sin backend interactivo: esto corre en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon

from src.config_loader import REPO_ROOT, load_config
from src.dashboard_data import (
    DISTRITOS_GEOJSON, PALETA_BANDAS, assign_bands_medias, load_distritos_geojson,
    load_instalaciones, load_output_csv, load_puntos, orden_bandas, orden_bandas_medias,
    split_access_population,
)
from src.metrics import weighted_mean_access

logger = logging.getLogger("fase5")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Handler propio: ver la nota en run_fase4.py sobre logging.basicConfig().
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] fase5: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

FIGURAS_DIR = REPO_ROOT / "report" / "figures"
TABLAS_DIR = REPO_ROOT / "report" / "tables"
DPI = 200

# Tipografía sobria y tamaño pensado para media página de un artículo a una
# columna; sin estilos de terceros para no añadir una dependencia por gusto.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
})

COLOR_DEP = {"PIURA": "#1b7837", "AYACUCHO": "#8c510a", "LORETO": "#01665e"}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_ESCAPES_LATEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}


def escapar_latex(texto) -> str:
    """Escapa los 12 caracteres que LaTeX interpreta. Sin esto, una banda
    llamada '>120' o una columna 'pct_del_total' rompen la compilación (el
    guión bajo entra en modo matemático y el '>' sale como símbolo raro)."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return "--"
    return "".join(_ESCAPES_LATEX.get(c, c) for c in str(texto))


def tabla_latex(
    df: pd.DataFrame, columnas: Sequence, encabezados: Sequence, alineacion: str,
    formatos: Optional[dict] = None, nombre: str = "tabla",
) -> Path:
    """Escribe un `tabular` booktabs en report/tables/{nombre}.tex.

    Se genera a mano en vez de con DataFrame.to_latex() para poder escapar
    el contenido, poner encabezados en castellano y envolver las columnas de
    texto largo en `\\makecell`/`p{}` sin pelearse con la API de pandas."""
    formatos = formatos or {}
    filas = []
    for _, fila in df.iterrows():
        celdas = []
        for col in columnas:
            valor = fila[col]
            fmt = formatos.get(col)
            if fmt and pd.notna(valor):
                celdas.append(escapar_latex(fmt.format(valor)) if isinstance(fmt, str) else escapar_latex(fmt(valor)))
            else:
                celdas.append(escapar_latex(valor))
        filas.append(" & ".join(celdas) + r" \\")

    tex = "\n".join([
        r"\begin{tabular}{" + alineacion + "}",
        r"\toprule",
        " & ".join(escapar_latex(h) for h in encabezados) + r" \\",
        r"\midrule",
        *filas,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    dest = TABLAS_DIR / f"{nombre}.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tex, encoding="utf-8")
    logger.info("Tabla: %s (%d filas)", _ruta_corta(dest), len(df))
    return dest


def _ruta_corta(path: Path) -> str:
    """Ruta relativa al repo para los logs, o absoluta si el destino está
    fuera (los tests escriben en un directorio temporal)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _miles(valor) -> str:
    return f"{float(valor):,.0f}"


def _un_decimal(valor) -> str:
    return f"{float(valor):,.1f}"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _guardar(fig, nombre: str) -> Path:
    """Guarda cada figura dos veces: PDF vectorial para el informe (la
    consigna pide vectorial o PNG de alta resolución, y el vectorial no se
    pixela al hacer zoom sobre un mapa) y PNG a 200 dpi para el README, el
    panel y el video, que no saben mostrar un PDF."""
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    dest_pdf = FIGURAS_DIR / f"{nombre}.pdf"
    dest_png = FIGURAS_DIR / f"{nombre}.png"
    fig.savefig(dest_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(dest_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Figura: %s (+ .png)", _ruta_corta(dest_pdf))
    return dest_pdf


def _anillos(geom: dict) -> list:
    """Anillos exteriores de una geometría GeoJSON (Polygon o MultiPolygon).
    Los anillos interiores (huecos) se ignoran: en los límites distritales
    peruanos no hay enclaves, y dibujarlos exigiría rutas compuestas."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]


def figura_mapa(cfg: dict, puntos: pd.DataFrame) -> Optional[Path]:
    """Coropleto distrital por banda de acceso + establecimientos resolutivos.

    Un panel por departamento, cada uno con su propio encuadre. Un solo mapa
    con los tres a la vez desperdicia media lámina en el vacío que hay entre
    Piura y Ayacucho, y deja los distritos costeros del tamaño de un sello.

    Se dibuja con matplotlib directamente sobre el GeoJSON recortado en vez
    de con geopandas: el informe no debería exigir la pila geoespacial
    completa solo para pintar 225 polígonos, y así esta figura se regenera en
    cualquier máquina que tenga el repo."""
    geojson = load_distritos_geojson(cfg)
    if geojson is None:
        logger.warning("Sin %s: se omite la figura del mapa.", DISTRITOS_GEOJSON)
        return None

    access_df, population_df = split_access_population(puntos)
    distritos = weighted_mean_access(access_df, population_df, level="distrito", cfg=cfg)
    banda_por_ubigeo = dict(zip(
        distritos["nivel_id"], assign_bands_medias(distritos["t_min_medio_ponderado"], cfg)
    ))

    instalaciones = load_instalaciones(cfg)
    resolutivas = instalaciones[instalaciones["es_resolutivo"].fillna(False)]

    # Orden costa -> sierra -> selva, el mismo con el que se justifica la
    # elección de departamentos en el informe.
    orden_dep = [(cfg["departamentos"][k]["ubigeo_dep"], cfg["departamentos"][k]["nombre"])
                 for k in ("costero", "andino", "amazonico")]
    por_dep = {ubigeo: [f for f in geojson["features"]
                        if f["properties"]["IDDIST"][:2] == ubigeo]
               for ubigeo, _ in orden_dep}

    # Ancho de cada panel proporcional a la relación de aspecto real del
    # departamento, para que los tres queden a la misma escala vertical y
    # ninguno salga estirado.
    extensiones = {}
    for ubigeo, feats in por_dep.items():
        coords = np.vstack([np.asarray(a) for f in feats for a in _anillos(f["geometry"])])
        extensiones[ubigeo] = (coords[:, 0].min(), coords[:, 0].max(),
                               coords[:, 1].min(), coords[:, 1].max())
    razones = [((e[1] - e[0]) / (e[3] - e[2])) for e in
               (extensiones[u] for u, _ in orden_dep)]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2),
                             gridspec_kw={"width_ratios": razones})
    sin_dato_total = 0
    for ax, (ubigeo, nombre) in zip(axes, orden_dep):
        parches, colores, sin_dato = [], [], 0
        for feat in por_dep[ubigeo]:
            banda = banda_por_ubigeo.get(feat["properties"]["IDDIST"])
            if banda is None:
                sin_dato += 1
            color = PALETA_BANDAS.get(banda, "#f0f0f0")
            for anillo in _anillos(feat["geometry"]):
                parches.append(MplPolygon(np.asarray(anillo), closed=True))
                colores.append(color)
        sin_dato_total += sin_dato

        ax.add_collection(PatchCollection(
            parches, facecolors=colores, edgecolors="white", linewidths=0.3, zorder=1,
        ))
        ipress_dep = resolutivas[resolutivas["DEPARTAMENTO"] == nombre]
        ax.scatter(ipress_dep["lon"], ipress_dep["lat"], s=22, c="#08306b", marker="o",
                   edgecolors="white", linewidths=0.5, zorder=3)

        lon_min, lon_max, lat_min, lat_max = extensiones[ubigeo]
        margen = 0.03 * max(lon_max - lon_min, lat_max - lat_min)
        ax.set_xlim(lon_min - margen, lon_max + margen)
        ax.set_ylim(lat_min - margen, lat_max + margen)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{nombre.title()}\n{len(ipress_dep)} IPRESS resolutivas", fontsize=10)

    handles = [Patch(facecolor=PALETA_BANDAS[b], edgecolor="white", label=_etiqueta_banda(b))
               for b in orden_bandas_medias(cfg)]
    handles.append(Patch(facecolor="#f0f0f0", edgecolor="white",
                         label=f"sin puntos muestreados ({sin_dato_total})"))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#08306b",
                          markeredgecolor="white", markersize=7,
                          label=f"IPRESS resolutiva ({len(resolutivas)})"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Tiempo medio de viaje en automóvil al establecimiento resolutivo más cercano\n"
        "(media distrital ponderada por población)", y=1.04, fontsize=11.5,
    )
    fig.tight_layout()
    return _guardar(fig, "fig_mapa_acceso")


def _etiqueta_banda(banda: str) -> str:
    return {
        "sin_acceso_enrutable": "sin ruta en la red vial",
        "sin_media_ponderable": "sin población censada",
    }.get(banda, f"{banda} min")


def figura_cobertura(cfg: dict, puntos: pd.DataFrame) -> Path:
    """Dos paneles: población por banda en el agregado y por departamento.

    El panel por departamento es el que cuenta la historia: el mismo país,
    tres realidades de acceso distintas."""
    from src.metrics import coverage_bands

    etiquetas = orden_bandas(cfg)
    global_ = load_output_csv("coverage_bands", cfg).set_index("banda")["pct_poblacion"]

    # Orden costa -> sierra -> selva (el de config.md), no alfabético: es el
    # eje sobre el que se eligieron los tres departamentos. barh apila de
    # abajo hacia arriba, así que se invierte para que Piura quede arriba.
    por_dep = {}
    for clave in ("amazonico", "andino", "costero"):
        dep = cfg["departamentos"][clave]["nombre"]
        sub = puntos[puntos["DEP"] == dep]
        if sub.empty:
            continue
        a, p = split_access_population(sub)
        por_dep[dep] = coverage_bands(a, p, cfg).set_index("banda")["pct_poblacion"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})

    ax1.bar(range(len(etiquetas)), [global_.get(b, 0) for b in etiquetas],
            color=[PALETA_BANDAS[b] for b in etiquetas], width=0.72)
    for i, b in enumerate(etiquetas):
        ax1.text(i, global_.get(b, 0) + 1.2, f"{global_.get(b, 0):.1f}%", ha="center", fontsize=9)
    ax1.set_xticks(range(len(etiquetas)))
    ax1.set_xticklabels([_etiqueta_banda(b).replace(" min", "").replace("sin ruta en la red vial", "sin\nruta")
                         for b in etiquetas], fontsize=9)
    ax1.set_ylabel("% de la población")
    ax1.set_ylim(0, 100)
    ax1.set_title("Los tres departamentos", loc="left")

    izquierda = np.zeros(len(por_dep))
    deps = list(por_dep)
    for b in etiquetas:
        valores = np.array([por_dep[d].get(b, 0.0) for d in deps])
        ax2.barh(deps, valores, left=izquierda, color=PALETA_BANDAS[b], height=0.6,
                 label=_etiqueta_banda(b))
        for y, (v, x0) in enumerate(zip(valores, izquierda)):
            if v >= 6:
                ax2.text(x0 + v / 2, y, f"{v:.0f}", ha="center", va="center",
                         fontsize=8.5, color="white", fontweight="bold")
        izquierda += valores
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% de la población del departamento")
    ax2.set_title("Por departamento", loc="left")
    ax2.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    return _guardar(fig, "fig_cobertura_bandas")


def figura_lorenz(cfg: dict) -> Path:
    lorenz = load_output_csv("gini_access_lorenz_curve", cfg)
    resumen = load_output_csv("gini_access_resumen", cfg)
    gini = float(resumen["gini"].iloc[0])

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#666", lw=1.2, label="Igualdad perfecta")
    ax.plot(lorenz["pct_poblacion_acumulada"], lorenz["pct_tiempo_acumulado"],
            color="#d73027", lw=2.2, label="Curva de Lorenz observada")
    ax.fill_between(lorenz["pct_poblacion_acumulada"], lorenz["pct_tiempo_acumulado"],
                    lorenz["pct_poblacion_acumulada"], color="#d73027", alpha=0.12)
    ax.set_xlabel("Proporción acumulada de población\n(ordenada de menor a mayor tiempo)")
    ax.set_ylabel("Proporción acumulada del tiempo total de viaje")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Concentración del tiempo de viaje (Gini = {gini:.3f})", loc="left")
    ax.legend(loc="upper left")
    return _guardar(fig, "fig_lorenz")


def figura_urbano_rural(cfg: dict) -> Path:
    urc = load_output_csv("urban_rural_contrast", cfg).set_index("zona")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    zonas = ["urbano", "rural"]
    valores = [float(urc.loc[z, "t_min_medio_ponderado"]) for z in zonas]
    ax.bar(zonas, valores, color=["#1a9850", "#d73027"], width=0.55)
    for x, v in zip(zonas, valores):
        ax.text(x, v + 5, f"{v:.0f} min", ha="center", fontsize=10)
    ax.set_ylabel("Minutos (media ponderada por población)")
    ax.set_ylim(0, max(valores) * 1.22)
    ax.set_title(f"Brecha rural/urbano: {valores[1] / valores[0]:.1f}×", loc="left")
    return _guardar(fig, "fig_urbano_rural")


def figura_ranking(cfg: dict) -> Path:
    ranking = load_output_csv("critical_gap_ranking", cfg).sort_values("t_min_medio_ponderado")
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    horas = ranking["t_min_medio_ponderado"] / 60
    ax.barh(ranking["nivel_nombre"], horas, color="#d73027", height=0.68)
    for y, (h, pob) in enumerate(zip(horas, ranking["poblacion_con_acceso"])):
        ax.text(h + max(horas) * 0.012, y, f"{h:.0f} h · {pob:,.0f} hab.",
                va="center", fontsize=8.5)
    ax.set_xlabel("Horas hasta el establecimiento resolutivo más cercano (media ponderada)")
    ax.set_xlim(0, max(horas) * 1.28)
    ax.set_title("Los 15 distritos con peor acceso (todos en Loreto)", loc="left")
    ax.grid(axis="y", visible=False)
    return _guardar(fig, "fig_ranking_criticos")


def figura_modos(cfg: dict) -> Path:
    modos = load_output_csv("comparacion_modos", cfg)
    datos = [modos[c].dropna() for c in ["min_car", "min_bike", "min_foot"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # `tick_labels` (matplotlib >= 3.9) reemplaza al viejo `labels`, que
    # matplotlib 3.11 ya eliminó.
    bp = ax.boxplot(datos, tick_labels=["automóvil", "bicicleta", "a pie"], showfliers=False,
                    patch_artist=True, widths=0.55)
    for parche, color in zip(bp["boxes"], ["#08306b", "#2b8cbe", "#a6bddb"]):
        parche.set_facecolor(color)
        parche.set_edgecolor("#333")
    for mediana in bp["medians"]:
        mediana.set_color("white")
        mediana.set_linewidth(1.8)
    ax.set_yscale("log")
    ax.set_ylabel("Minutos al establecimiento resolutivo (escala log)")
    ax.set_title(
        "Mismo origen y mismo destino, tres modos\n"
        f"(mediana a pie / automóvil: {modos['ratio_foot_car'].median():.1f}×)", loc="left",
    )
    ax.grid(axis="x", visible=False)
    return _guardar(fig, "fig_modos")


def figura_altitud(cfg: dict, puntos: pd.DataFrame) -> Path:
    """Dispersión altitud vs. tiempo, coloreada por departamento.

    El color no es decorativo: es la evidencia visual de por qué la
    correlación global es engañosa. Los peores tiempos están en Loreto, a
    menos de 200 msnm, no en las cumbres de Ayacucho."""
    resumen = load_output_csv("access_vs_altitude_resumen", cfg)
    rho = float(resumen["spearman_rho"].iloc[0])
    datos = puntos.dropna(subset=["t_min", "Z"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for dep, color in COLOR_DEP.items():
        sub = datos[datos["DEP"] == dep]
        ax.scatter(sub["Z"], sub["t_min"], s=9, alpha=0.35, color=color,
                   label=f"{dep} (n={len(sub):,})", linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("Altitud del centro poblado (msnm)")
    ax.set_ylabel("Minutos al establecimiento resolutivo (log)")
    ax.set_title(f"Altitud y tiempo de acceso (Spearman global $\\rho$ = {rho:.3f})", loc="left")
    leyenda = ax.legend(loc="upper right", fontsize=9)
    for handle in leyenda.legend_handles:
        handle.set_alpha(1)
    return _guardar(fig, "fig_altitud")


def figura_recta_vs_red(cfg: dict) -> Path:
    """Distancia de red frente a distancia en línea recta (Discusión).

    Es la figura que justifica por qué no se puede medir accesibilidad con
    una regla sobre el mapa: la nube se separa de la diagonal de forma
    distinta en cada departamento, y en Loreto se separa además hacia arriba
    en tiempo, no solo en distancia."""
    pares = load_output_csv("calibracion_pares_muestra", cfg)
    calib = load_output_csv("calibracion_linea_recta", cfg).set_index("departamento")
    factor_config = cfg["enrutamiento"]["fallback_no_enrutable"]["factor_desvio_default"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    for dep, color in COLOR_DEP.items():
        sub = pares[pares["DEP"] == dep]
        ax1.scatter(sub["dist_recta_km"], sub["distancia_km"], s=5, alpha=0.18,
                    color=color, linewidths=0, label=dep)
    # Ejes logarítmicos: la matriz OD es demanda x TODAS las resolutivas, así
    # que los pares van de menos de 1 km a más de 1,000. En escala lineal, el
    # 90 % de la nube se aplasta contra el origen.
    piso, tope = 0.5, float(pares["dist_recta_km"].max()) * 1.1
    recta = np.array([piso, tope])
    ax1.plot(recta, recta, color="#333", lw=1.2, ls="--", label="línea recta (factor 1.0)")
    ax1.plot(recta, recta * float(calib.loc["TODOS", "factor_desvio_mediano"]),
             color="#d73027", lw=1.6,
             label=f"factor mediano observado ({calib.loc['TODOS', 'factor_desvio_mediano']:.2f})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(piso, tope)
    ax1.set_ylim(piso, tope * 2)
    ax1.set_xlabel("Distancia en línea recta (km, escala log)")
    ax1.set_ylabel("Distancia por la red vial (km, escala log)")
    ax1.set_title("Cada par origen-destino enrutado", loc="left")
    leyenda = ax1.legend(fontsize=8.5, loc="lower right")
    for handle in leyenda.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1)

    deps = [d for d in COLOR_DEP if d in calib.index]
    x = np.arange(len(deps))
    ax2.bar(x - 0.2, [calib.loc[d, "factor_desvio_mediano"] for d in deps], width=0.38,
            color=[COLOR_DEP[d] for d in deps], label="Factor de desvío (red / recta)")
    ax2.axhline(factor_config, color="#d73027", ls="--", lw=1.3)
    ax2.text(-0.45, 2.3, f"- - -  config.md usaba {factor_config} por defecto",
             ha="left", va="top", fontsize=8.5, color="#d73027")
    for i, d in enumerate(deps):
        ax2.text(i - 0.2, calib.loc[d, "factor_desvio_mediano"] + 0.03,
                 f"{calib.loc[d, 'factor_desvio_mediano']:.2f}", ha="center", fontsize=9)

    ax3 = ax2.twinx()
    ax3.bar(x + 0.2, [calib.loc[d, "velocidad_mediana_kmh"] for d in deps], width=0.38,
            color="#bdbdbd", edgecolor="#666")
    for i, d in enumerate(deps):
        ax3.text(i + 0.2, calib.loc[d, "velocidad_mediana_kmh"] + 1.5,
                 f"{calib.loc[d, 'velocidad_mediana_kmh']:.0f} km/h", ha="center", fontsize=8.5)
    ax3.set_ylabel("Velocidad mediana de la red (km/h)")
    ax3.set_ylim(0, 85)
    ax3.grid(False)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.title() for d in deps])
    ax2.set_ylabel("Factor de desvío (veces la línea recta)")
    ax2.set_ylim(0, 2.4)
    ax2.set_title("Desvío y velocidad, por departamento", loc="left")
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    return _guardar(fig, "fig_recta_vs_red")


# ---------------------------------------------------------------------------
# Tablas del informe
# ---------------------------------------------------------------------------

def tablas(cfg: dict) -> None:
    bandas = load_output_csv("coverage_bands", cfg)
    bandas = bandas.assign(banda_legible=bandas["banda"].map(_etiqueta_banda))
    tabla_latex(
        bandas, ["banda_legible", "poblacion", "pct_poblacion"],
        ["Banda de acceso", "Población estimada", "% de la población"], "lrr",
        {"poblacion": _miles, "pct_poblacion": lambda v: f"{v:.2f}"},
        nombre="informe_cobertura_bandas",
    )

    deps = load_output_csv("weighted_mean_access_departamento", cfg)
    tabla_latex(
        deps, ["nivel_nombre", "t_min_medio_ponderado", "poblacion_con_acceso",
               "poblacion_sin_acceso_enrutable", "pct_poblacion_con_acceso", "n_puntos"],
        ["Departamento", "Tiempo medio (min)", "Población con ruta",
         "Población sin ruta", "% con ruta", "Puntos"], "lrrrrr",
        {"t_min_medio_ponderado": _un_decimal, "poblacion_con_acceso": _miles,
         "poblacion_sin_acceso_enrutable": _miles,
         "pct_poblacion_con_acceso": lambda v: f"{v:.1f}", "n_puntos": _miles},
        nombre="informe_departamentos",
    )

    ranking = load_output_csv("critical_gap_ranking", cfg).head(10)
    ranking = ranking.assign(horas=ranking["t_min_medio_ponderado"] / 60)
    tabla_latex(
        ranking, ["ranking", "nivel_nombre", "nivel_id", "horas", "poblacion_con_acceso"],
        # Encabezados en texto plano: tabla_latex() los escapa, así que aquí
        # no se pueden colar comandos LaTeX (saldrían impresos literalmente).
        ["Puesto", "Distrito", "Ubigeo", "Horas", "Población"], "rllrr",
        {"horas": _un_decimal, "poblacion_con_acceso": _miles},
        nombre="informe_ranking",
    )

    urc = load_output_csv("urban_rural_contrast", cfg)
    tabla_latex(
        urc, ["zona", "t_min_medio_ponderado", "poblacion_con_acceso",
              "poblacion_sin_acceso_enrutable", "n_puntos"],
        ["Zona", "Tiempo medio (min)", "Población con ruta", "Población sin ruta", "Puntos"],
        "lrrrr",
        {"t_min_medio_ponderado": _un_decimal, "poblacion_con_acceso": _miles,
         "poblacion_sin_acceso_enrutable": _miles, "n_puntos": _miles},
        nombre="informe_urbano_rural",
    )

    for origen, nombre in [("data_quality_report", "informe_calidad_oferta"),
                           ("data_quality_report_demanda", "informe_calidad_demanda")]:
        calidad = load_output_csv(origen, cfg)
        # La justificación completa es un párrafo por regla: va en el anexo
        # del informe en prosa, no en una celda de tabla.
        calidad = calidad.assign(accion_corta=calidad["accion"].str.split("(").str[0].str.strip())
        tabla_latex(
            calidad, ["regla", "n_marcados", "pct_del_total", "accion_corta"],
            ["Regla", "Marcados", "% del total", "Acción"], "lrrp{5.4cm}",
            {"n_marcados": _miles, "pct_del_total": lambda v: f"{v:.2f}"},
            nombre=nombre,
        )

    snap = load_output_csv("snapping_report", cfg)
    tabla_latex(
        snap, ["dataset", "perfil", "n_puntos", "n_fallidos", "pct_fallidos",
               "distancia_media_snap_m"],
        ["Conjunto", "Perfil", "Puntos", "Sin red vial", "% sin red vial", "Distancia media (m)"],
        "llrrrr",
        {"n_puntos": _miles, "n_fallidos": _miles, "pct_fallidos": lambda v: f"{v:.1f}",
         "distancia_media_snap_m": _un_decimal},
        nombre="informe_snapping",
    )

    calib = load_output_csv("calibracion_linea_recta", cfg)
    tabla_latex(
        calib, ["departamento", "n_pares", "factor_desvio_p25", "factor_desvio_mediano",
                "factor_desvio_p75", "velocidad_mediana_kmh"],
        ["Departamento", "Pares enrutados", "Desvío P25", "Desvío mediano", "Desvío P75",
         "Velocidad mediana (km/h)"], "lrrrrr",
        {"n_pares": _miles, "factor_desvio_p25": lambda v: f"{v:.2f}",
         "factor_desvio_mediano": lambda v: f"{v:.2f}",
         "factor_desvio_p75": lambda v: f"{v:.2f}",
         "velocidad_mediana_kmh": _un_decimal},
        nombre="informe_calibracion",
    )

    modos = load_output_csv("comparacion_modos", cfg)
    resumen_modos = pd.DataFrame([
        {"modo": etiqueta,
         "n": int(modos[col].notna().sum()),
         "mediana": modos[col].median(),
         "p75": modos[col].quantile(0.75),
         "ratio": modos[ratio].median() if ratio else np.nan}
        for col, etiqueta, ratio in [
            ("min_car", "Automóvil", None),
            ("min_bike", "Bicicleta", "ratio_bike_car"),
            ("min_foot", "A pie", "ratio_foot_car"),
        ]
    ])
    tabla_latex(
        resumen_modos, ["modo", "n", "mediana", "p75", "ratio"],
        ["Modo", "Puntos", "Mediana (min)", "P75 (min)", "Veces el tiempo en automóvil"],
        "lrrrr",
        {"n": _miles, "mediana": _un_decimal, "p75": _un_decimal,
         # Sin "$\\times$": el contenido de las celdas se escapa (es dato, no
         # marcado), así que un comando LaTeX aquí saldría impreso tal cual.
         "ratio": lambda v: f"{v:.1f}"},
        nombre="informe_modos",
    )


# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    puntos = load_puntos(cfg)

    figura_mapa(cfg, puntos)
    figura_cobertura(cfg, puntos)
    figura_lorenz(cfg)
    figura_urbano_rural(cfg)
    figura_ranking(cfg)
    figura_modos(cfg)
    figura_altitud(cfg, puntos)
    figura_recta_vs_red(cfg)
    tablas(cfg)

    figuras = sorted(p.stem for p in FIGURAS_DIR.glob("*.pdf"))
    tablas_generadas = sorted(p.name for p in TABLAS_DIR.glob("informe_*.tex"))
    print("\n--- Fase 5 ---")
    print(f"Figuras ({len(figuras)}): {', '.join(figuras)}")
    print(f"Tablas  ({len(tablas_generadas)}): {', '.join(tablas_generadas)}")
    print("\nCompila el informe con:  cd report && pdflatex informe.tex && pdflatex informe.tex")


if __name__ == "__main__":
    main()
