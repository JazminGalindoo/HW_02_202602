"""
dashboard_data.py — Fase 4: capa de carga de datos del panel
==============================================================

Separación de responsabilidades (mismo criterio que metrics.py / export.py):

    metrics.py          calcula (funciones puras, sin I/O)
    export.py           escribe los outputs de Fase 3
    dashboard_data.py   LEE esos artefactos y los deja en la forma exacta
                        que esperan las funciones de metrics.py
    app.py              solo dibuja (Streamlit / Plotly)

Este módulo NO implementa ninguna métrica. Cuando el usuario del panel
filtra por departamento o por zona, el panel vuelve a llamar a las MISMAS
funciones de `src/metrics.py` sobre el subconjunto filtrado -- no hay una
segunda implementación de "media ponderada" o "bandas de cobertura" viviendo
en la capa de presentación, que es justamente el error que produce paneles
que no cuadran con el informe.

Tampoco importa `streamlit`: así es testeable con pytest sin levantar la app
(ver tests/test_dashboard.py). El cacheo (`st.cache_data`) se aplica en
`app.py`, envolviendo estas funciones.

Artefactos que consume (los genera `run_fase4.py`, ver su docstring):
    data/processed/puntos_dashboard.parquet        1 fila por centro poblado
    data/processed/instalaciones_dashboard.parquet 1 fila por IPRESS
    data/processed/distritos_3dep.geojson          polígonos de los 3 deptos
    data/processed/mejoras_candidatos.parquet      insumo del simulador
    data/outputs/*.csv                             salidas de Fase 1/2/3
    logs/acquisition_summary.json                  trazabilidad de descargas
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.config_loader import REPO_ROOT, load_config

logger = logging.getLogger("dashboard_data")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Ver la nota equivalente en metrics.py sobre por qué no usamos
    # logging.basicConfig() aquí.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] dashboard_data: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


# Rutas de los artefactos propios de Fase 4. No van a config.md > rutas
# porque config.md es la fuente de verdad de PARÁMETROS del pipeline; estos
# son derivados internos del panel y se regeneran con run_fase4.py.
PUNTOS_DASHBOARD = "data/processed/puntos_dashboard.parquet"
INSTALACIONES_DASHBOARD = "data/processed/instalaciones_dashboard.parquet"
DISTRITOS_GEOJSON = "data/processed/distritos_3dep.geojson"
MEJORAS_CANDIDATOS = "data/processed/mejoras_candidatos.parquet"
CALIBRACION_RECTA = "data/outputs/calibracion_linea_recta.csv"
CALIBRACION_PARES = "data/outputs/calibracion_pares_muestra.csv"
RESUMEN_FASE4 = "data/outputs/resumen_fase4.json"

# Columnas que produce access_time() y que, por tanto, NO deben viajar
# dentro del population_df (metrics._join_by_id las descartaría del merge
# silenciosamente y el resultado sería confuso de depurar).
_COLS_ACCESO = ["t_min", "id_instalacion", "flag_sin_acceso_enrutable"]
_COLS_SOLO_PANEL = ["en_muestra_enrutada", "banda"]

# Paleta de las bandas de acceso. Verde -> rojo, con un gris explícito para
# "sin acceso enrutable": esa categoría no es "peor que >120 min", es
# *desconocida* (no hay ruta en la red mapeada), y merece un color que no la
# haga parecer parte del gradiente.
PALETA_BANDAS = {
    "0-30": "#1a9850",
    "30-60": "#a6d96a",
    "60-120": "#fdae61",
    ">120": "#d73027",
    "sin_acceso_enrutable": "#9e9e9e",
    # Solo aplica a agregados (ver assign_bands_medias): una unidad cuyo
    # promedio no se puede calcular NO es una unidad con mal acceso.
    "sin_media_ponderable": "#e0e0e0",
}

BANDA_SIN_MEDIA = "sin_media_ponderable"


# ---------------------------------------------------------------------------
# Resolución de rutas
# ---------------------------------------------------------------------------

def artefacto(rel_path: str) -> Path:
    """Ruta absoluta de un artefacto relativo a la raíz del repo."""
    return REPO_ROOT / rel_path


def _exigir(path: Path, como_generarlo: str) -> Path:
    """Falla con un mensaje accionable en vez de un FileNotFoundError pelado:
    el evaluador que clona el repo y corre el panel sin haber ejecutado la
    preparación debe leer QUÉ comando le falta, no un traceback."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genéralo con: {como_generarlo}"
        )
    return path


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------

def load_puntos(cfg: Optional[dict] = None) -> pd.DataFrame:
    """Tabla larga de demanda: 1 fila por centro poblado del universo
    (~19,460), con población, zona urbano/rural, peso de muestreo y -- para
    los que entraron a la muestra enrutada de Fase 2 -- t_min y banda."""
    cfg = cfg or load_config()
    path = _exigir(artefacto(PUNTOS_DASHBOARD), "python run_fase4.py")
    df = pd.read_parquet(path)
    logger.info("puntos_dashboard: %d filas, %d en la muestra enrutada.",
                len(df), int(df["en_muestra_enrutada"].sum()))
    return df


def load_instalaciones(cfg: Optional[dict] = None) -> pd.DataFrame:
    """Tabla de oferta: 1 fila por IPRESS con coordenada válida, con
    categoría normalizada y la bandera es_resolutivo de Fase 1."""
    cfg = cfg or load_config()
    path = _exigir(artefacto(INSTALACIONES_DASHBOARD), "python run_fase4.py")
    return pd.read_parquet(path)


def load_distritos_geojson(cfg: Optional[dict] = None) -> Optional[dict]:
    """GeoJSON recortado a los 3 departamentos, para el coropleto.

    Devuelve None (no lanza) si el archivo no está: el panel debe seguir
    funcionando con el mapa de puntos aunque falte el recorte de polígonos
    -- los límites distritales son la única fuente del pipeline que se
    descarga en tiempo de ejecución, y el panel no puede depender de que el
    portal esté arriba el día de la evaluación."""
    path = artefacto(DISTRITOS_GEOJSON)
    if not path.exists():
        logger.warning(
            "No existe %s; el panel omitirá el coropleto y usará solo el mapa de puntos. "
            "Genéralo con: python run_fase4.py", DISTRITOS_GEOJSON,
        )
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_mejoras_candidatos(cfg: Optional[dict] = None) -> pd.DataFrame:
    """Pares (centro poblado, establecimiento candidato a ascenso) en los que
    ascender ese establecimiento mejoraría el tiempo actual. Insumo del
    simulador de escenarios; ver `run_fase4.construir_mejoras_candidatos`
    para el origen de los tiempos."""
    cfg = cfg or load_config()
    path = _exigir(artefacto(MEJORAS_CANDIDATOS), "python run_fase4.py")
    df = pd.read_parquet(path)
    logger.info("mejoras_candidatos: %d pares sobre %d establecimientos candidatos (fuente: %s).",
                len(df), df["id_candidato"].nunique(),
                "/".join(sorted(df["fuente"].unique())) if len(df) else "-")
    return df


def load_output_csv(nombre: str, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Lee data/outputs/{nombre}.csv (salidas ya calculadas de Fase 1/2/3).

    El panel las usa tal cual para las vistas NO filtradas (el informe cita
    exactamente estos números). Las vistas filtradas se recalculan con
    metrics.py sobre el subconjunto -- ver el aviso en la barra lateral."""
    cfg = cfg or load_config()
    path = _exigir(
        artefacto(f"{cfg['rutas']['outputs']}/{nombre}.csv"),
        "python run_fase1.py && python run_fase2.py && python run_fase3.py",
    )
    return pd.read_csv(path)


def load_acquisition_summary(cfg: Optional[dict] = None) -> dict:
    """logs/acquisition_summary.json: qué fuente se descargó y cuál falló.

    Se muestra en la pestaña de calidad de datos porque las dos fuentes que
    fallaron (RENIPRESS y SIGMED) son un hallazgo del trabajo, no un detalle
    de plomería que convenga esconder."""
    cfg = cfg or load_config()
    path = artefacto(f"{cfg['rutas']['logs']}/acquisition_summary.json")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_resumen_fase4(cfg: Optional[dict] = None) -> dict:
    """Metadatos de la corrida de run_fase4.py (fecha, conteos, artefactos
    de origen) para la pestaña de reproducibilidad."""
    path = artefacto(RESUMEN_FASE4)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Adaptación al contrato de metrics.py
# ---------------------------------------------------------------------------

def split_access_population(puntos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte la tabla del panel en los dos DataFrames que esperan las
    funciones de metrics.py: (access_df, population_df).

    access_df SOLO contiene los puntos que entraron a la muestra enrutada de
    Fase 2. Esto es crítico y no cosmético: si se incluyeran los ~14,500
    puntos del universo que nunca se enrutaron, su t_min NaN los haría
    contarse como 'sin acceso enrutable' y el panel reportaría una mayoría de
    población sin acceso que es puro artefacto del muestreo.

    population_df conserva TODAS las filas (con peso_muestral NaN -> peso 0
    para las no muestreadas), que es exactamente lo que recibió Fase 3."""
    faltantes = [
        c for c in ["id", "en_muestra_enrutada", "t_min", "flag_sin_acceso_enrutable"]
        if c not in puntos.columns
    ]
    if faltantes:
        raise KeyError(f"split_access_population: faltan columnas en puntos: {faltantes}")

    en_muestra = puntos["en_muestra_enrutada"].fillna(False).astype(bool)
    access_df = (
        puntos.loc[en_muestra, ["id", "t_min", "flag_sin_acceso_enrutable"]]
        .rename(columns={"id": "id_demanda"})
        .reset_index(drop=True)
    )
    access_df["flag_sin_acceso_enrutable"] = (
        access_df["flag_sin_acceso_enrutable"].fillna(False).astype(bool)
    )

    population_df = puntos.drop(
        columns=[c for c in _COLS_ACCESO + _COLS_SOLO_PANEL if c in puntos.columns]
    )
    return access_df, population_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bandas de acceso (presentación)
# ---------------------------------------------------------------------------

def band_labels(cfg: Optional[dict] = None) -> list:
    """Etiquetas de banda en orden, con los mismos bordes que
    metrics.coverage_bands (config.md > metricas.bandas_acceso_min)."""
    cfg = cfg or load_config()
    bordes = cfg["metricas"]["bandas_acceso_min"]
    return (
        [f"0-{bordes[0]}"]
        + [f"{bordes[i]}-{bordes[i + 1]}" for i in range(len(bordes) - 1)]
        + [f">{bordes[-1]}"]
    )


def assign_bands(t_min: pd.Series, cfg: Optional[dict] = None) -> pd.Series:
    """Etiqueta de banda por punto, para COLOREAR el mapa.

    Ojo: los porcentajes de población por banda que muestra el panel NO
    salen de aquí, salen de metrics.coverage_bands() (ponderados por
    población y peso muestral). Esta función solo reparte puntos en
    categorías de color y comparte los bordes con aquella para que el mapa y
    la tabla no se contradigan. t_min NaN -> 'sin_acceso_enrutable'."""
    cfg = cfg or load_config()
    bordes = cfg["metricas"]["bandas_acceso_min"]
    etiquetas = band_labels(cfg)
    bins = [-1e-9] + list(bordes) + [np.inf]
    banda = pd.cut(t_min, bins=bins, labels=etiquetas)
    return banda.astype("object").where(t_min.notna(), "sin_acceso_enrutable")


def assign_bands_medias(medias: pd.Series, cfg: Optional[dict] = None) -> pd.Series:
    """Banda de una MEDIA agregada (distrito, provincia, departamento).

    Difiere de assign_bands en el significado del NaN, y la diferencia no es
    cosmética. En un punto, t_min NaN significa ``no hay ruta''. En una media
    ponderada, NaN significa que la unidad no tiene peso poblacional: todos
    sus centros poblados quedaron sin población censada conocida (ver el
    reporte de cruce con el Censo 2017). Son cosas distintas y pintarlas del
    mismo gris haría leer como ``sin acceso'' a distritos urbanos como
    Veintiséis de Octubre (Piura) o San Juan Bautista (Ayacucho), que tienen
    ruta calculada para todos sus puntos muestreados."""
    cfg = cfg or load_config()
    banda = assign_bands(medias, cfg)
    return banda.where(medias.notna(), BANDA_SIN_MEDIA)


def orden_bandas(cfg: Optional[dict] = None) -> list:
    """Orden canónico para leyendas y ejes (bandas + la de sin acceso)."""
    return band_labels(cfg) + ["sin_acceso_enrutable"]


def orden_bandas_medias(cfg: Optional[dict] = None) -> list:
    """Orden canónico para leyendas de mapas y tablas de agregados."""
    return band_labels(cfg) + [BANDA_SIN_MEDIA]


# ---------------------------------------------------------------------------
# Filtros del panel
# ---------------------------------------------------------------------------

def filter_puntos(
    puntos: pd.DataFrame,
    departamentos: Optional[Iterable] = None,
    provincias: Optional[Iterable] = None,
    zonas: Optional[Iterable] = None,
    solo_muestra_enrutada: bool = False,
) -> pd.DataFrame:
    """Filtra la tabla de demanda por departamento, provincia y/o zona.

    None = 'sin filtro' (no es lo mismo que una lista vacía, que sí filtra a
    cero filas y así el usuario ve explícitamente que deseleccionó todo).

    El filtro de provincia usa el nombre (columna PROV) porque es lo que el
    usuario elige en pantalla; los nombres de provincia se repiten entre
    departamentos, así que el filtro de departamento debe aplicarse antes ---
    como se hace aquí --- para que 'HUANCABAMBA' no arrastre provincias
    homónimas de otra región."""
    out = puntos
    if departamentos is not None:
        out = out[out["DEP"].isin(list(departamentos))]
    if provincias is not None:
        out = out[out["PROV"].isin(list(provincias))]
    if zonas is not None:
        out = out[out["zona"].isin(list(zonas))]
    if solo_muestra_enrutada:
        out = out[out["en_muestra_enrutada"].fillna(False).astype(bool)]
    return out.reset_index(drop=True)


def filter_instalaciones(
    instalaciones: pd.DataFrame,
    departamentos: Optional[Iterable] = None,
    provincias: Optional[Iterable] = None,
    categorias: Optional[Iterable] = None,
    instituciones: Optional[Iterable] = None,
    solo_resolutivas: bool = False,
) -> pd.DataFrame:
    """Mismo criterio de filtrado para la capa de oferta del mapa, más los
    dos ejes que solo tienen sentido en la oferta: categoría (I-1 ... III-E,
    incluida SIN_CATEGORIA) e institución (MINSA, EsSalud, privados,
    Gobierno Regional, sanidades...)."""
    out = instalaciones
    if departamentos is not None:
        out = out[out["DEPARTAMENTO"].isin(list(departamentos))]
    if provincias is not None:
        out = out[out["PROVINCIA"].isin(list(provincias))]
    if categorias is not None:
        out = out[out["categoria_norm"].isin(list(categorias))]
    if instituciones is not None:
        out = out[out["INSTITUCION"].isin(list(instituciones))]
    if solo_resolutivas:
        out = out[out["es_resolutivo"].fillna(False).astype(bool)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

def formato_minutos(valor: Any) -> str:
    """Minutos legibles: 47 min, 3.4 h, 2.3 días. Los tiempos de Loreto
    llegan a ~4,900 minutos y '4909.07' no le dice nada a nadie."""
    if valor is None:
        return "s/d"
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return "s/d"
    if np.isnan(valor):
        return "s/d"
    if valor < 90:
        return f"{valor:.0f} min"
    if valor < 60 * 24:
        return f"{valor / 60:.1f} h"
    return f"{valor / (60 * 24):.1f} días"


def etiquetas_legibles(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas técnicas a castellano para las tablas del panel.
    Solo presentación: el CSV que se descarga conserva los nombres originales
    del pipeline, para que sea el mismo archivo que cita el informe."""
    mapa = {
        "nivel_id": "Ubigeo",
        "nivel_nombre": "Unidad",
        "t_min_medio_ponderado": "t_min medio ponderado (min)",
        "poblacion_con_acceso": "Población con acceso enrutable",
        "poblacion_sin_acceso_enrutable": "Población sin acceso enrutable",
        "pct_poblacion_con_acceso": "% población con acceso",
        "n_puntos": "N.º de centros poblados",
        "banda": "Banda de acceso (min)",
        "poblacion": "Población",
        "pct_poblacion": "% de población",
        "zona": "Zona",
        "ranking": "#",
        "regla": "Regla",
        "n_marcados": "Registros marcados",
        "accion": "Acción",
        "justificacion": "Justificación",
        "pct_del_total": "% del total",
        "perfil": "Perfil",
        "n_puntos_snap": "Puntos evaluados",
        "n_fallidos": "Puntos sin red vial cercana",
        "pct_fallidos": "% sin red vial cercana",
        "distancia_media_snap_m": "Distancia media al eje vial (m)",
        "distancia_max_snap_m": "Distancia máxima al eje vial (m)",
        "dataset": "Conjunto",
    }
    return df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})
