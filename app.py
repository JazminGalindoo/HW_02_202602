"""
app.py — Fase 4: panel interactivo (Streamlit)
================================================

Levantar con:

    pip install -r requirements.txt
    python run_fase4.py          # una vez, prepara los artefactos
    streamlit run app.py

No requiere OSRM, ni Docker, ni geopandas: todo lo que muestra sale de los
parquet/CSV que dejaron las Fases 1-3 y `run_fase4.py`. El panel **no llama a
ningún motor de enrutamiento ni reconstruye un grafo**; con los artefactos ya
generados abre en un par de segundos.

Principio de diseño (el mismo que declara metrics.py): **ninguna métrica se
implementa aquí**. Cuando el usuario filtra por departamento, provincia o
zona, el panel vuelve a llamar a `src.metrics.*` sobre el subconjunto
filtrado. Si alguien cambia la definición de una banda de cobertura, del Gini
o del escenario de ascenso, el panel y el informe cambian juntos, porque leen
la misma función. Esta capa solo:
    (a) carga artefactos (vía src/dashboard_data.py),
    (b) filtra,
    (c) dibuja.

Sobre la honestidad de las cifras: el panel repite en cada vista de qué
universo habla. Las métricas se calculan sobre la muestra estratificada de
5,002 centros poblados que se logró enrutar en Fase 2 (de 19,460), ponderada
por población censada 2017 y por el peso de diseño del muestreo. No se
presenta ningún número como si describiera a los tres departamentos completos
sin ese matiz.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app.py` ejecuta este archivo como script suelto; añadir la
# raíz del repo a sys.path permite `import src...` sin exigir `pip install -e .`
# ni un PYTHONPATH manual al evaluador.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import dashboard_data as dd
from src.config_loader import load_config
from src.metrics import (
    access_vs_altitude, apply_upgrades, coverage_bands, critical_gap_ranking, gini_access,
    population_within, ranking_upgrade_gain, urban_rural_contrast, weighted_mean_access,
    weighted_median_access,
)

st.set_page_config(
    page_title="Golden Hour — acceso a salud resolutiva",
    page_icon="🏥",
    layout="wide",
)

CFG = load_config()
ORDEN_BANDAS = dd.orden_bandas(CFG)
ORDEN_BANDAS_MEDIAS = dd.orden_bandas_medias(CFG)
PALETA = dict(dd.PALETA_BANDAS)
PALETA["no_muestreado"] = "#d9d9d9"

# Centro aproximado de los tres departamentos (Piura al NO, Ayacucho al S,
# Loreto al NE): el punto medio del bounding box conjunto, no un valor
# elegido a ojo.
CENTRO_MAPA = {"lat": -7.5, "lon": -76.0}
ZOOM_MAPA = 4.3

# Umbral fijo del indicador "población desatendida" que pide la consigna,
# independiente del umbral que el usuario mueva en la barra lateral.
UMBRAL_DESATENCION_MIN = 60

# ---------------------------------------------------------------------------
# Compatibilidad Plotly 5 (Mapbox) / 6+ (MapLibre)
# ---------------------------------------------------------------------------
# plotly 6 renombró todas las trazas de mapa (scattermapbox -> scattermap) y
# eliminó las viejas en la 7. requirements.txt pide plotly>=5.20, así que el
# panel tiene que funcionar en ambas: se detecta la API disponible una vez.
_ES_MAPLIBRE = hasattr(go, "Scattermap")
_TrazaPuntos = go.Scattermap if _ES_MAPLIBRE else go.Scattermapbox
_ESTILO_MAPA = "carto-positron"  # sin token, disponible en ambas APIs

# streamlit 1.49 deprecó `use_container_width=True` en favor de
# `width="stretch"`. requirements.txt pide streamlit>=1.35, así que se elige
# el argumento vigente en tiempo de ejecución en vez de llenar la consola del
# evaluador de avisos de deprecación (o de romperse en una versión vieja).
_VERSION_ST = tuple(int(x) for x in st.__version__.split(".")[:2])
_ANCHO = {"width": "stretch"} if _VERSION_ST >= (1, 49) else {"use_container_width": True}


def _px_mapa(fn_nueva: str, fn_vieja: str, df, **kwargs):
    """Llama a la función de plotly.express que exista, pasando el nombre de
    argumento de estilo correcto para esa versión."""
    if hasattr(px, fn_nueva):
        return getattr(px, fn_nueva)(df, map_style=_ESTILO_MAPA, **kwargs)
    return getattr(px, fn_vieja)(df, mapbox_style=_ESTILO_MAPA, **kwargs)


def _etiqueta_ipress(df: pd.DataFrame) -> pd.Series:
    """Texto del tooltip de la capa de oferta. `categoria_norm` puede ser
    'SIN_CATEGORIA' o nula (390 establecimientos del registro no traen una
    categoría legible): se muestra tal cual en vez de dejar el tooltip en
    blanco, porque esa ausencia es información."""
    categoria = df["categoria_norm"].fillna("sin categoría")
    institucion = df["INSTITUCION"].fillna("institución no declarada")
    return df["NOMBRE"].fillna("(sin nombre)") + " · " + categoria + " · " + institucion


def _layout_mapa(fig, alto: int = 620):
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=alto,
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "xanchor": "left", "x": 0.01,
                "bgcolor": "rgba(255,255,255,0.75)"},
    )
    return fig


# ---------------------------------------------------------------------------
# Carga cacheada (el cacheo vive aquí, no en dashboard_data.py, para que ese
# módulo sea importable y testeable sin Streamlit)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando centros poblados…")
def cargar_puntos() -> pd.DataFrame:
    return dd.load_puntos(CFG)


@st.cache_data(show_spinner="Cargando establecimientos…")
def cargar_instalaciones() -> pd.DataFrame:
    return dd.load_instalaciones(CFG)


@st.cache_data(show_spinner="Cargando límites distritales…")
def cargar_geojson():
    return dd.load_distritos_geojson(CFG)


@st.cache_data(show_spinner="Cargando escenarios…")
def cargar_mejoras() -> pd.DataFrame:
    return dd.load_mejoras_candidatos(CFG)


@st.cache_data
def cargar_output(nombre: str) -> pd.DataFrame:
    return dd.load_output_csv(nombre, CFG)


@st.cache_data
def cargar_json(cual: str) -> dict:
    return dd.load_acquisition_summary(CFG) if cual == "adquisicion" else dd.load_resumen_fase4(CFG)


@st.cache_data(show_spinner="Agregando por distrito…")
def agregado_distrital(deps: tuple, provs: tuple, zonas: tuple) -> pd.DataFrame:
    """Media ponderada a nivel distrito, siempre (el coropleto es distrital
    aunque el usuario esté mirando las tablas a nivel provincia)."""
    sub = dd.filter_puntos(cargar_puntos(), departamentos=deps, provincias=provs, zonas=zonas)
    access_df, population_df = dd.split_access_population(sub)
    return weighted_mean_access(access_df, population_df, level="distrito", cfg=CFG)


@st.cache_data(show_spinner="Recalculando métricas sobre el filtro…")
def metricas(deps: tuple, provs: tuple, zonas: tuple, nivel: str, umbral: int) -> dict:
    """Recalcula TODAS las métricas del panel sobre el subconjunto filtrado,
    llamando a las funciones de src/metrics.py (no reimplementándolas).

    Se cachea por la tupla de filtros: volver a un filtro ya visto es
    instantáneo. Devuelve un dict de DataFrames para hacer una sola pasada
    por el subconjunto en vez de nueve."""
    sub = dd.filter_puntos(cargar_puntos(), departamentos=deps, provincias=provs, zonas=zonas)
    if sub.empty:
        return {"vacio": True, "n_universo": 0, "n_muestra": 0}

    access_df, population_df = dd.split_access_population(sub)
    if access_df.empty:
        return {"vacio": True, "n_universo": len(sub), "n_muestra": 0}

    agregado = weighted_mean_access(access_df, population_df, level=nivel, cfg=CFG)
    distrital = (agregado if nivel == "distrito"
                 else weighted_mean_access(access_df, population_df, level="distrito", cfg=CFG))
    peor = distrital.dropna(subset=["t_min_medio_ponderado"])
    gini_resumen, lorenz = gini_access(access_df, population_df)
    altitud_puntos, altitud_resumen = access_vs_altitude(access_df, population_df)

    return {
        "vacio": False,
        "n_universo": len(sub),
        "n_muestra": len(access_df),
        "bandas": coverage_bands(access_df, population_df, CFG),
        "agregado": agregado,
        "gini": gini_resumen,
        "lorenz": lorenz,
        "contraste": urban_rural_contrast(access_df, population_df),
        "altitud_puntos": altitud_puntos,
        "altitud_resumen": altitud_resumen,
        "mediana": weighted_median_access(access_df, population_df),
        "umbral": population_within(access_df, population_df, umbral),
        "desatencion": population_within(access_df, population_df, UMBRAL_DESATENCION_MIN),
        "peor_distrito": peor.iloc[0].to_dict() if len(peor) else None,
        "puntos_filtrados": sub[["id", "DEP", "PROV", "DIST", "NOMCP", "zona", "lat", "lon",
                                 "t_min", "banda", "poblacion_censada", "Z",
                                 "en_muestra_enrutada"]],
    }


@st.cache_data(show_spinner="Simulando el escenario…")
def simular(deps: tuple, provs: tuple, zonas: tuple, umbral: int, ascendidos: tuple) -> dict:
    """Aplica el ascenso de los establecimientos seleccionados y devuelve el
    antes y el después. Usa metrics.apply_upgrades + metrics.population_within
    + metrics.coverage_bands: las mismas funciones del resto del panel, de
    modo que la cobertura del escenario es comparable con la de la pestaña
    de resumen."""
    sub = dd.filter_puntos(cargar_puntos(), departamentos=deps, provincias=provs, zonas=zonas)
    access_df, population_df = dd.split_access_population(sub)
    if access_df.empty:
        return {"vacio": True}

    mejoras = cargar_mejoras()
    mejoras = mejoras[mejoras["id_demanda"].isin(set(access_df["id_demanda"]))]

    antes = population_within(access_df, population_df, umbral)
    nuevo_access = apply_upgrades(access_df, mejoras, ascendidos)
    despues = population_within(nuevo_access, population_df, umbral)

    beneficiados = nuevo_access[nuevo_access["mejorado"]]
    detalle = (
        population_df[population_df["id"].isin(set(beneficiados["id_demanda"]))]
        .groupby(["DEP", "DIST"], as_index=False)
        .agg(centros_poblados=("id", "size"), poblacion=("poblacion_censada", "sum"))
        .sort_values("poblacion", ascending=False)
    )

    return {
        "vacio": False,
        "antes": antes,
        "despues": despues,
        "bandas_antes": coverage_bands(access_df, population_df, CFG),
        "bandas_despues": coverage_bands(nuevo_access, population_df, CFG),
        "n_puntos_mejorados": int(nuevo_access["mejorado"].sum()),
        "detalle_distritos": detalle,
        "ranking": ranking_upgrade_gain(access_df, population_df, mejoras, umbral, top_n=15),
    }


# ---------------------------------------------------------------------------
# Barra lateral: filtros
# ---------------------------------------------------------------------------

puntos_all = cargar_puntos()
instalaciones_all = cargar_instalaciones()
deps_disponibles = sorted(puntos_all["DEP"].dropna().unique().tolist())
categorias_disponibles = sorted(instalaciones_all["categoria_norm"].dropna().unique().tolist())
instituciones_disponibles = sorted(instalaciones_all["INSTITUCION"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filtros")
    deps = st.multiselect("Departamento", deps_disponibles, default=deps_disponibles)

    provs_disponibles = sorted(
        puntos_all.loc[puntos_all["DEP"].isin(deps), "PROV"].dropna().unique().tolist()
    )
    provs_sel = st.multiselect(
        "Provincia", provs_disponibles, default=[],
        help="Vacío = todas las provincias de los departamentos seleccionados.",
    )
    provs = provs_sel or None  # lista vacía = sin filtro, no "cero provincias"

    zonas = st.multiselect(
        "Zona", ["urbano", "rural"], default=["urbano", "rural"],
        help="Proxy administrativo: 'urbano' = centro poblado capital de distrito, "
             "provincia o departamento (CAPITAL != 0). No es la definición oficial "
             "de área urbana del INEI; ver la pestaña Metodología.",
    )
    umbral = st.slider(
        "Umbral de acceso (minutos)", min_value=15, max_value=240,
        value=int(CFG["metricas"]["bandas_acceso_min"][0]), step=15,
        help="Define qué cuenta como 'población cubierta' en los indicadores y en el "
             "simulador de escenarios.",
    )

    st.divider()
    st.caption("**Capa de establecimientos** (mapa y simulador)")
    categorias = st.multiselect(
        "Categoría", categorias_disponibles, default=categorias_disponibles,
        help="SIN_CATEGORIA = el registro no trae una categoría legible (390 casos).",
    )
    instituciones = st.multiselect(
        "Institución", instituciones_disponibles, default=instituciones_disponibles,
        help="GOBIERNO REGIONAL es la red pública descentralizada (la mayor parte de lo "
             "que coloquialmente se llama 'MINSA'); la etiqueta MINSA queda para los "
             "pocos establecimientos de administración central.",
    )

    st.divider()
    nivel = st.selectbox(
        "Nivel de agregación de las tablas", ["distrito", "provincia", "departamento"], index=0,
        help="Se agrupa por prefijo de ubigeo (6/4/2 dígitos), no por nombre de texto.",
    )
    st.caption(
        "Al mover un filtro, el panel **recalcula** las métricas llamando a las mismas "
        "funciones de `src/metrics.py` que produjeron las tablas del informe. "
        "Con todo seleccionado, los números coinciden exactamente con `data/outputs/`."
    )
    st.caption(
        f"Universo: **{len(puntos_all):,}** centros poblados · "
        f"muestra enrutada en Fase 2: **{int(puntos_all['en_muestra_enrutada'].sum()):,}**"
    )

deps_t = tuple(deps)
provs_t = tuple(provs) if provs else None
zonas_t = tuple(zonas)
M = metricas(deps_t, provs_t, zonas_t, nivel, umbral)

st.title("Golden Hour — acceso por carretera a salud resolutiva")
st.markdown(
    "Tiempo de viaje en automóvil desde cada centro poblado hasta el establecimiento "
    "de salud **con capacidad resolutiva** (categoría II-1 o superior, RENIPRESS) más "
    "cercano, en **Piura** (costa), **Ayacucho** (sierra) y **Loreto** (selva)."
)

if M.get("vacio"):
    st.warning(
        "El filtro actual no deja ningún centro poblado con ruta calculada. "
        "Selecciona al menos un departamento y una zona (y revisa el filtro de provincia)."
    )
    st.stop()

instalaciones_filtradas = dd.filter_instalaciones(
    instalaciones_all, departamentos=deps, provincias=provs,
    categorias=categorias, instituciones=instituciones,
)


def _pct_banda(bandas: pd.DataFrame, etiqueta: str) -> float:
    fila = bandas.loc[bandas["banda"] == etiqueta, "pct_poblacion"]
    return float(fila.iloc[0]) if len(fila) else float("nan")


# ---------------------------------------------------------------------------
# Encabezado de indicadores (se recalcula con cada filtro)
# ---------------------------------------------------------------------------

peor = M["peor_distrito"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    f"Población cubierta (≤ {umbral} min)",
    f"{M['umbral']['pct_bajo_umbral']:.1f} %",
    help=f"{M['umbral']['poblacion_bajo_umbral']:,.0f} de "
         f"{M['umbral']['poblacion_total']:,.0f} habitantes estimados en el filtro actual.",
)
k2.metric(
    f"Población a más de {UMBRAL_DESATENCION_MIN} min",
    f"{M['desatencion']['poblacion_sobre_umbral']:,.0f}",
    f"{M['desatencion']['pct_sobre_umbral']:.1f} % de la población",
    delta_color="inverse",
)
k3.metric(
    "Mediana del tiempo de acceso", dd.formato_minutos(M["mediana"]),
    help="Ponderada por población: el tiempo de la persona que está justo en el medio. "
         "Muy por debajo de la media, que arrastra la cola amazónica.",
)
k4.metric(
    "Peor distrito", peor["nivel_nombre"] if peor else "s/d",
    dd.formato_minutos(peor["t_min_medio_ponderado"]) if peor else None,
    delta_color="off",
)
k5.metric(
    "Gini del acceso", f"{float(M['gini']['gini'].iloc[0]):.3f}",
    help="0 = todos esperan lo mismo; 1 = el tiempo total de viaje se concentra en una minoría.",
)

tabs = st.tabs([
    "Resumen", "Mapa", "Distribución", "Brechas", "Simulador de escenarios",
    "Equidad", "Modos de viaje", "Altitud", "Calidad de datos", "Metodología",
])
(tab_resumen, tab_mapa, tab_dist, tab_brechas, tab_sim,
 tab_equidad, tab_modos, tab_altitud, tab_calidad, tab_metodo) = tabs


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    bandas = M["bandas"]
    st.subheader("Población por banda de tiempo de acceso")
    col_g, col_t = st.columns([3, 2])
    with col_g:
        fig = px.bar(
            bandas, x="banda", y="pct_poblacion", color="banda",
            color_discrete_map=PALETA, category_orders={"banda": ORDEN_BANDAS},
            labels={"banda": "Banda (minutos)", "pct_poblacion": "% de la población"},
            text=bandas["pct_poblacion"].map(lambda v: f"{v:.1f} %"),
        )
        fig.update_layout(showlegend=False, height=380, yaxis_title="% de la población")
        st.plotly_chart(fig, **_ANCHO)
    with col_t:
        st.dataframe(
            dd.etiquetas_legibles(bandas).style.format(
                {"Población": "{:,.0f}", "% de población": "{:.2f}"}),
            hide_index=True, **_ANCHO,
        )
        st.caption(
            "`sin_acceso_enrutable` no es 'más de 120 minutos': son centros poblados para los que "
            "**no existe ninguna ruta** hacia un establecimiento resolutivo en la red vial de "
            "OpenStreetMap. En Loreto eso refleja que el acceso real es fluvial, no vial."
        )

    st.subheader("Contraste urbano / rural")
    contraste = M["contraste"]
    cols = st.columns(len(contraste) + 1)
    for i, fila in contraste.reset_index(drop=True).iterrows():
        cols[i].metric(
            f"Tiempo medio — {fila['zona']}", dd.formato_minutos(fila["t_min_medio_ponderado"]),
            help=f"{fila['n_puntos']:,} centros poblados en la muestra filtrada.",
        )
    if {"urbano", "rural"}.issubset(set(contraste["zona"])):
        t_u = float(contraste.loc[contraste["zona"] == "urbano", "t_min_medio_ponderado"].iloc[0])
        t_r = float(contraste.loc[contraste["zona"] == "rural", "t_min_medio_ponderado"].iloc[0])
        cols[-1].metric("Brecha rural / urbano", f"{t_r / t_u:.1f}×" if t_u else "s/d")

    st.info(
        f"Estas cifras describen los **{M['n_muestra']:,}** centros poblados con ruta calculada "
        f"(de **{M['n_universo']:,}** en el filtro actual), ponderados por población censada 2017 "
        "y por el peso de diseño del muestreo estratificado de Fase 2. No son un censo de los tres "
        "departamentos completos."
    )


# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------
with tab_mapa:
    c1, c2, c3 = st.columns([2, 2, 3])
    vista = c1.radio("Vista", ["Coropleto por distrito", "Centros poblados"],
                     horizontal=False, label_visibility="collapsed")
    capa_ipress = c2.radio(
        "Establecimientos", ["Solo resolutivos", "Todos", "Ninguno"],
        horizontal=False, label_visibility="collapsed",
    )
    if vista == "Centros poblados":
        mostrar_no_muestreados = c3.checkbox(
            "Incluir centros poblados fuera de la muestra enrutada (sin t_min)", value=False)
    else:
        mostrar_no_muestreados = False
        c3.caption(
            "La capa de establecimientos respeta los filtros de categoría e institución de la "
            "barra lateral."
        )

    geojson = cargar_geojson()
    if capa_ipress == "Ninguno":
        capa = instalaciones_filtradas.iloc[0:0]
    elif capa_ipress == "Solo resolutivos":
        capa = instalaciones_filtradas[instalaciones_filtradas["es_resolutivo"].fillna(False)]
    else:
        capa = instalaciones_filtradas

    resolutivas_capa = capa[capa["es_resolutivo"].fillna(False)]
    otras_capa = capa[~capa["es_resolutivo"].fillna(False)]

    def _agregar_capa_ipress(fig):
        """Dos trazas separadas --- resolutivas y no resolutivas --- para que
        la leyenda permita apagar cada una y para que no se confundan en el
        mapa: son categorías con significado clínico distinto, no un degradado."""
        if len(otras_capa):
            fig.add_trace(_TrazaPuntos(
                lat=otras_capa["lat"], lon=otras_capa["lon"], mode="markers",
                marker={"size": 6, "color": "#9ecae1"},
                name=f"IPRESS no resolutiva ({len(otras_capa):,})",
                text=_etiqueta_ipress(otras_capa), hovertemplate="%{text}<extra></extra>",
            ))
        if len(resolutivas_capa):
            fig.add_trace(_TrazaPuntos(
                lat=resolutivas_capa["lat"], lon=resolutivas_capa["lon"], mode="markers",
                marker={"size": 11, "color": "#08306b"},
                name=f"IPRESS resolutiva ({len(resolutivas_capa)})",
                text=_etiqueta_ipress(resolutivas_capa), hovertemplate="%{text}<extra></extra>",
            ))
        return fig

    if vista == "Coropleto por distrito":
        if geojson is None:
            st.error(
                "Falta `data/processed/distritos_3dep.geojson`. Genéralo con `python run_fase4.py` "
                "(descarga los límites distritales si no están en `data/raw/`)."
            )
        else:
            # Se colorea por BANDA del tiempo medio distrital, no por una
            # escala continua: con Loreto en ~4,900 min y Piura en ~5 min,
            # cualquier gradiente continuo deja a los otros dos
            # departamentos indistinguibles en un solo tono.
            distritos = agregado_distrital(deps_t, provs_t, zonas_t).assign(
                banda=lambda d: dd.assign_bands_medias(d["t_min_medio_ponderado"], CFG),
                t_legible=lambda d: d["t_min_medio_ponderado"].map(dd.formato_minutos),
            )
            fig = _px_mapa(
                "choropleth_map", "choropleth_mapbox", distritos,
                geojson=geojson, locations="nivel_id", color="banda",
                color_discrete_map=PALETA, category_orders={"banda": ORDEN_BANDAS_MEDIAS},
                hover_name="nivel_nombre",
                hover_data={"nivel_id": True, "t_legible": True, "banda": False,
                            "poblacion_con_acceso": ":,.0f", "n_puntos": True},
                labels={"t_legible": "Tiempo medio", "nivel_id": "Ubigeo",
                        "poblacion_con_acceso": "Población con acceso", "n_puntos": "Centros poblados"},
                center=CENTRO_MAPA, zoom=ZOOM_MAPA, opacity=0.75,
            )
            st.plotly_chart(_layout_mapa(_agregar_capa_ipress(fig)), **_ANCHO)
            n_sin_media = int((distritos["banda"] == dd.BANDA_SIN_MEDIA).sum())
            st.caption(
                f"{len(distritos)} distritos con al menos un centro poblado muestreado. "
                "Los distritos en blanco no tienen ningún punto enrutado en el filtro actual "
                "(o su polígono viene vacío en la fuente de límites, ver Limitaciones). "
                f"Los {n_sin_media} distritos en gris claro sí tienen ruta calculada, pero "
                "ninguno de sus centros poblados empató con el Censo 2017, así que no admiten "
                "media ponderada por población: gris no significa mal acceso, significa sin dato "
                "de población."
            )
    else:
        pts = M["puntos_filtrados"]
        if not mostrar_no_muestreados:
            pts = pts[pts["en_muestra_enrutada"].fillna(False)]
        pts = pts.assign(t_legible=pts["t_min"].map(dd.formato_minutos))
        orden = ORDEN_BANDAS + (["no_muestreado"] if mostrar_no_muestreados else [])
        fig = _px_mapa(
            "scatter_map", "scatter_mapbox", pts,
            lat="lat", lon="lon", color="banda",
            color_discrete_map=PALETA, category_orders={"banda": orden},
            hover_name="NOMCP",
            hover_data={"t_legible": True, "DIST": True, "zona": True,
                        "poblacion_censada": ":,.0f", "lat": False, "lon": False, "banda": False},
            labels={"t_legible": "Tiempo al hospital", "DIST": "Distrito",
                    "poblacion_censada": "Población 2017"},
            center=CENTRO_MAPA, zoom=ZOOM_MAPA,
        )
        fig.update_traces(marker={"size": 6})
        st.plotly_chart(_layout_mapa(_agregar_capa_ipress(fig)), **_ANCHO)
        st.caption(
            f"{len(pts):,} centros poblados dibujados · {len(capa):,} establecimientos en la capa. "
            "Un punto gris es un centro poblado que no entró al muestreo de Fase 2: no tiene "
            "tiempo calculado, que no es lo mismo que no tener acceso."
        )


# ---------------------------------------------------------------------------
# Distribución
# ---------------------------------------------------------------------------
with tab_dist:
    st.subheader("Distribución del tiempo de acceso")
    corte = st.radio("Desagregar por", ["departamento", "zona (urbano/rural)", "sin desagregar"],
                     horizontal=True)
    col_corte = {"departamento": "DEP", "zona (urbano/rural)": "zona"}.get(corte)

    pts = M["puntos_filtrados"]
    pts = pts[pts["en_muestra_enrutada"].fillna(False)].dropna(subset=["t_min"])

    if pts.empty:
        st.warning("El filtro actual no deja puntos con tiempo calculado.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.ecdf(
                pts, x="t_min", color=col_corte, log_x=True,
                labels={"t_min": "Minutos al establecimiento resolutivo (escala log)",
                        "DEP": "Departamento", "zona": "Zona"},
                color_discrete_sequence=px.colors.qualitative.Dark2,
            )
            fig.add_vline(x=umbral, line_dash="dash", line_color="#666",
                          annotation_text=f"umbral {umbral} min", annotation_position="top left")
            fig.update_layout(height=430, yaxis_title="Proporción acumulada de centros poblados")
            st.plotly_chart(fig, **_ANCHO)
            st.caption(
                "ECDF: la altura de la curva sobre el umbral es la proporción de **centros "
                "poblados** por debajo de ese tiempo. Ojo: aquí cada punto pesa igual; los "
                "porcentajes ponderados por población están en el encabezado y en Resumen."
            )
        with col_b:
            fig2 = px.histogram(
                pts, x="t_min", color=col_corte, nbins=60, log_x=True, barmode="overlay",
                opacity=0.65,
                labels={"t_min": "Minutos (escala log)", "DEP": "Departamento", "zona": "Zona"},
                color_discrete_sequence=px.colors.qualitative.Dark2,
            )
            fig2.add_vline(x=umbral, line_dash="dash", line_color="#666")
            fig2.update_layout(height=430, yaxis_title="Centros poblados")
            st.plotly_chart(fig2, **_ANCHO)
            st.caption(
                "La bimodalidad de Loreto no es ruido: hay un grupo cerca de la capital "
                "provincial y otro a días de viaje, sin nada en medio."
            )

        resumen = pts.groupby(col_corte)["t_min"].describe() if col_corte else pts["t_min"].describe().to_frame().T
        st.dataframe(
            resumen[["count", "25%", "50%", "75%", "max"]].rename(columns={
                "count": "Centros poblados", "25%": "P25 (min)", "50%": "Mediana (min)",
                "75%": "P75 (min)", "max": "Máximo (min)"}).style.format("{:,.0f}"),
            **_ANCHO,
        )


# ---------------------------------------------------------------------------
# Brechas
# ---------------------------------------------------------------------------
with tab_brechas:
    agregado = M["agregado"]
    st.subheader(f"Tiempo medio ponderado por {nivel}")

    top_n = st.slider("Unidades a mostrar en el gráfico", 5, 40,
                      min(15, max(5, len(agregado))), step=5)
    peores = agregado.dropna(subset=["t_min_medio_ponderado"]).head(top_n).iloc[::-1]
    fig = px.bar(
        peores, x="t_min_medio_ponderado", y="nivel_nombre", orientation="h",
        color="t_min_medio_ponderado", color_continuous_scale="Reds",
        labels={"t_min_medio_ponderado": "Minutos (media ponderada por población)",
                "nivel_nombre": ""},
        hover_data={"nivel_id": True, "poblacion_con_acceso": ":,.0f", "n_puntos": True},
    )
    fig.update_layout(height=max(360, 22 * len(peores)), coloraxis_showscale=False)
    st.plotly_chart(fig, **_ANCHO)

    st.dataframe(
        dd.etiquetas_legibles(agregado).style.format({
            "t_min medio ponderado (min)": "{:,.1f}",
            "Población con acceso enrutable": "{:,.0f}",
            "Población sin acceso enrutable": "{:,.0f}",
            "% población con acceso": "{:.1f}",
        }),
        hide_index=True, height=420, **_ANCHO,
    )
    st.caption(
        "La tabla es ordenable: haz clic en cualquier encabezado. "
        "`% población con acceso` expone qué parte de la población de esa unidad quedó **fuera** "
        "del promedio por no tener ruta calculada; una unidad con tiempo medio bajo y 30 % de "
        "cobertura no es una unidad bien servida."
    )
    st.download_button(
        f"⬇️ Descargar tabla por {nivel} (CSV)", agregado.to_csv(index=False).encode("utf-8"),
        file_name=f"acceso_por_{nivel}.csv", mime="text/csv",
    )

    st.subheader("Ranking de brechas críticas")
    c1, c2 = st.columns(2)
    n_rank = c1.number_input("Tamaño del ranking", 5, 50,
                             int(CFG["metricas"]["ranking_criticos"]["n"]), step=5)
    pob_min = c2.number_input(
        "Población mínima para entrar al ranking", 0, 50_000,
        int(CFG["metricas"]["ranking_criticos"]["poblacion_minima"]), step=100,
        help="Evita que una unidad de poquísimos habitantes con un tiempo altísimo domine el "
             "ranking. El valor por defecto viene de config.md y el aplicado se guarda en el CSV.",
    )
    ranking = critical_gap_ranking(agregado, n=int(n_rank), poblacion_minima=float(pob_min), cfg=CFG)
    st.dataframe(
        dd.etiquetas_legibles(ranking.drop(columns=["poblacion_minima_aplicada"])).style.format({
            "t_min medio ponderado (min)": "{:,.1f}",
            "Población con acceso enrutable": "{:,.0f}",
            "Población sin acceso enrutable": "{:,.0f}",
            "% población con acceso": "{:.1f}",
        }),
        hide_index=True, **_ANCHO,
    )
    st.download_button(
        "⬇️ Descargar ranking (CSV)", ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_brechas_{nivel}.csv", mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Simulador de escenarios
# ---------------------------------------------------------------------------
with tab_sim:
    st.subheader("¿Qué pasaría si se ascendiera un establecimiento a capacidad resolutiva?")
    st.markdown(
        "Selecciona uno o más establecimientos **I-3 o I-4** existentes y el panel recalcula la "
        "cobertura suponiendo que pasan a resolver emergencias. El tiempo de cada centro poblado "
        "se vuelve el mínimo entre su hospital actual y el establecimiento ascendido más cercano "
        f"(`metrics.apply_upgrades`), y la ganancia se mide con el umbral de **{umbral} minutos** "
        "de la barra lateral."
    )

    mejoras = cargar_mejoras()
    fuente = "/".join(sorted(mejoras["fuente"].unique())) if len(mejoras) else "-"
    if fuente == "estimado":
        st.warning(
            "**Los tiempos hacia los candidatos son estimados, no enrutados.** La matriz OD de "
            "Fase 2 se calculó contra los 58 establecimientos resolutivos, no contra los 607 "
            "candidatos a ascenso. Mientras no exista "
            "`data/processed/matriz_car_candidatos.parquet` (ver README), el simulador estima "
            "esos tiempos con la calibración empírica línea-recta contra red de cada "
            "departamento. Sirve para comparar y priorizar candidatos; no para prometer un "
            "tiempo concreto a un centro poblado concreto.",
            icon="⚠️",
        )
    else:
        st.success("Los tiempos hacia los candidatos vienen de la matriz OSRM demanda × candidatos.")

    candidatos = instalaciones_filtradas[
        instalaciones_filtradas["categoria_norm"].isin(["I-3", "I-4"])
    ].copy()
    candidatos = candidatos[candidatos["COD_IPRESS"].isin(set(mejoras["id_candidato"]))]

    if candidatos.empty:
        st.info(
            "Ningún establecimiento I-3/I-4 en el filtro actual. Revisa los filtros de "
            "categoría e institución en la barra lateral."
        )
    else:
        candidatos = candidatos.assign(
            etiqueta=candidatos["NOMBRE"].fillna("(sin nombre)") + " — "
            + candidatos["categoria_norm"] + " · " + candidatos["DISTRITO"].fillna("")
            + " (" + candidatos["DEPARTAMENTO"].fillna("") + ")"
        )
        etiqueta_a_id = dict(zip(candidatos["etiqueta"], candidatos["COD_IPRESS"]))

        sim_previo = simular(deps_t, provs_t, zonas_t, umbral, ())
        sugeridos = sim_previo.get("ranking", pd.DataFrame())
        if len(sugeridos):
            sugeridos = sugeridos.merge(
                candidatos[["COD_IPRESS", "etiqueta", "DEPARTAMENTO", "DISTRITO", "categoria_norm"]],
                left_on="id_candidato", right_on="COD_IPRESS", how="inner",
            )
            st.markdown("**Candidatos con mayor ganancia individual** (elige de aquí o busca abajo)")
            st.dataframe(
                sugeridos[["etiqueta", "poblacion_ganada", "n_puntos_ganados"]].rename(columns={
                    "etiqueta": "Establecimiento",
                    "poblacion_ganada": f"Población que entraría a ≤ {umbral} min",
                    "n_puntos_ganados": "Centros poblados beneficiados",
                }).style.format({f"Población que entraría a ≤ {umbral} min": "{:,.0f}"}),
                hide_index=True, height=240, **_ANCHO,
            )
            st.caption(
                "Ganancias **individuales**: no se suman. Dos establecimientos vecinos cubren en "
                "buena medida a la misma gente, así que el efecto conjunto es menor que la suma "
                "de sus filas — selecciónalos abajo para ver el efecto real de la combinación."
            )

        seleccion = st.multiselect(
            "Establecimientos a ascender a categoría resolutiva",
            options=sorted(candidatos["etiqueta"]),
            default=list(sugeridos["etiqueta"].head(3)) if len(sugeridos) else [],
        )
        ascendidos = tuple(sorted(etiqueta_a_id[e] for e in seleccion))

        S = simular(deps_t, provs_t, zonas_t, umbral, ascendidos)
        if S.get("vacio"):
            st.warning("El filtro actual no deja puntos con ruta calculada.")
        else:
            antes, despues = S["antes"], S["despues"]
            ganancia = despues["poblacion_bajo_umbral"] - antes["poblacion_bajo_umbral"]
            ganancia_pct = despues["pct_bajo_umbral"] - antes["pct_bajo_umbral"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Cobertura actual (≤ {umbral} min)", f"{antes['pct_bajo_umbral']:.1f} %",
                      help=f"{antes['poblacion_bajo_umbral']:,.0f} habitantes.")
            c2.metric("Cobertura en el escenario", f"{despues['pct_bajo_umbral']:.1f} %",
                      f"{ganancia_pct:+.1f} pp")
            c3.metric("Ganancia marginal (población)", f"{ganancia:,.0f}",
                      help="Población que pasa a estar dentro del umbral gracias a los ascensos.")
            c4.metric("Centros poblados que mejoran", f"{S['n_puntos_mejorados']:,}")

            comparacion = (
                S["bandas_antes"][["banda", "pct_poblacion"]].rename(columns={"pct_poblacion": "Actual"})
                .merge(S["bandas_despues"][["banda", "pct_poblacion"]]
                       .rename(columns={"pct_poblacion": "Escenario"}), on="banda")
            )
            largo = comparacion.melt(id_vars="banda", var_name="Situación", value_name="pct")
            fig = px.bar(
                largo, x="banda", y="pct", color="Situación", barmode="group",
                category_orders={"banda": ORDEN_BANDAS},
                color_discrete_sequence=["#9ecae1", "#08306b"],
                labels={"banda": "Banda (minutos)", "pct": "% de la población"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, **_ANCHO)

            if not ascendidos:
                st.info("Selecciona al menos un establecimiento para ver el efecto del escenario.")
            elif len(S["detalle_distritos"]):
                st.markdown("**Dónde se siente la mejora**")
                st.dataframe(
                    S["detalle_distritos"].rename(columns={
                        "DEP": "Departamento", "DIST": "Distrito",
                        "centros_poblados": "Centros poblados que mejoran",
                        "poblacion": "Población beneficiada",
                    }).style.format({"Población beneficiada": "{:,.0f}"}),
                    hide_index=True, height=280, **_ANCHO,
                )
                st.download_button(
                    "⬇️ Descargar el escenario (CSV)",
                    S["detalle_distritos"].to_csv(index=False).encode("utf-8"),
                    file_name="escenario_ascensos.csv", mime="text/csv",
                )


# ---------------------------------------------------------------------------
# Equidad
# ---------------------------------------------------------------------------
with tab_equidad:
    gini_resumen = M["gini"]
    lorenz = M["lorenz"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Gini del tiempo de acceso", f"{float(gini_resumen['gini'].iloc[0]):.3f}")
    c2.metric("Puntos considerados", f"{int(gini_resumen['n_puntos'].iloc[0]):,}")
    c3.metric("Excluidos sin ruta", f"{int(gini_resumen['n_excluidos_sin_acceso'].iloc[0]):,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lorenz["pct_poblacion_acumulada"], y=lorenz["pct_tiempo_acumulado"],
        mode="lines", name="Curva de Lorenz", line={"color": "#d73027", "width": 3},
        fill="tozeroy", fillcolor="rgba(215,48,39,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Igualdad perfecta",
        line={"color": "#666", "dash": "dash"},
    ))
    fig.update_layout(
        height=460,
        xaxis_title="Proporción acumulada de población (de menor a mayor tiempo)",
        yaxis_title="Proporción acumulada del tiempo total de viaje",
        legend={"x": 0.02, "y": 0.98},
    )
    st.plotly_chart(fig, **_ANCHO)
    st.markdown(
        "**Cómo leerla.** Si el acceso fuera parejo, la curva sería la diagonal. Aquí la curva "
        "se pega al eje: la mayoría de la población acumula una fracción mínima del tiempo total "
        "de viaje, y ese tiempo se concentra en una minoría rural. Un Gini alto **no** significa "
        "'mal acceso promedio', significa acceso **desigualmente repartido**: dos territorios con "
        "el mismo tiempo medio pueden tener Gini muy distinto."
    )

    st.subheader("Urbano vs. rural")
    contraste = M["contraste"]
    fig2 = px.bar(
        contraste, x="zona", y="t_min_medio_ponderado", color="zona",
        color_discrete_map={"urbano": "#1a9850", "rural": "#d73027"},
        labels={"zona": "", "t_min_medio_ponderado": "Minutos (media ponderada)"},
        text=contraste["t_min_medio_ponderado"].map(dd.formato_minutos),
    )
    fig2.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig2, **_ANCHO)
    st.dataframe(
        dd.etiquetas_legibles(contraste).style.format({
            "t_min medio ponderado (min)": "{:,.1f}",
            "Población con acceso enrutable": "{:,.0f}",
            "Población sin acceso enrutable": "{:,.0f}",
        }),
        hide_index=True, **_ANCHO,
    )
    st.caption(
        "Clasificación urbano/rural = proxy administrativo (capital de distrito, provincia o "
        "departamento). Solo 237 de 19,460 centros poblados son capital de algo, así que 'urbano' "
        "aquí es un grupo chico y muy poblado; ver Metodología."
    )


# ---------------------------------------------------------------------------
# Modos de viaje
# ---------------------------------------------------------------------------
with tab_modos:
    st.subheader("Auto, bicicleta y a pie hacia el establecimiento resolutivo más cercano")
    ids_filtro = {str(i) for i in M["puntos_filtrados"]["id"]}

    modos = cargar_output("comparacion_modos")
    modos = modos[modos["id_demanda"].astype(str).isin(ids_filtro)]

    if modos.empty:
        st.warning("Ningún punto del filtro actual tiene los tres modos calculados.")
    else:
        largo = modos.melt(
            id_vars="id_demanda", value_vars=["min_car", "min_bike", "min_foot"],
            var_name="modo", value_name="minutos",
        ).replace({"modo": {"min_car": "auto", "min_bike": "bicicleta", "min_foot": "a pie"}})
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                largo.dropna(subset=["minutos"]), x="modo", y="minutos", color="modo",
                log_y=True, labels={"minutos": "Minutos (escala logarítmica)", "modo": ""},
                color_discrete_sequence=["#08306b", "#2b8cbe", "#a6bddb"],
            )
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, **_ANCHO)
            st.caption(
                "Escala logarítmica: sin ella, los tiempos a pie de Loreto (días enteros) aplastan "
                "toda la distribución del auto contra el eje."
            )
        with c2:
            resumen_modos = largo.groupby("modo")["minutos"].describe()[["count", "50%", "75%", "max"]]
            resumen_modos.columns = ["N.º de puntos", "Mediana (min)", "P75 (min)", "Máximo (min)"]
            st.dataframe(resumen_modos.style.format("{:,.0f}"), **_ANCHO)
            ratio = modos["ratio_foot_car"].dropna()
            if len(ratio):
                st.metric("Mediana del cociente a pie / auto", f"{ratio.median():.1f}×")
            ratio_b = modos["ratio_bike_car"].dropna()
            if len(ratio_b):
                st.metric("Mediana del cociente bicicleta / auto", f"{ratio_b.median():.1f}×")

    st.subheader("¿Cambia el establecimiento más cercano según el modo?")
    st.markdown(
        "Comparación exigida por la consigna, sobre los centros poblados **urbanos**: a pie se "
        "consideran **todos** los establecimientos (incluidos los de categoría I, que resuelven "
        "atención primaria), en auto solo los **resolutivos**. Si el establecimiento más cercano "
        "cambia con el modo, la política de 'centro de salud más cercano' depende de cómo se viaja."
    )
    pie_coche = cargar_output("comparacion_pie_vs_coche")
    pie_coche = pie_coche[pie_coche["id_demanda"].astype(str).isin(ids_filtro)]
    if pie_coche.empty:
        st.info(
            "El filtro actual no incluye centros poblados urbanos: esta comparación se calculó "
            "solo sobre los 237 centros poblados capital. Activa la zona 'urbano' para verla."
        )
    else:
        cambia = int((~pie_coche["misma_instalacion"].fillna(False)).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Puntos urbanos comparados", f"{len(pie_coche):,}")
        c2.metric("Cambia el establecimiento más cercano", f"{cambia:,}",
                  f"{100 * cambia / len(pie_coche):.0f} % de los puntos")
        c3.metric("Sin ruta peatonal", f"{int(pie_coche['min_foot'].isna().sum()):,}")
        fig = px.histogram(
            pie_coche.dropna(subset=["ratio_foot_car"]), x="ratio_foot_car", nbins=40,
            labels={"ratio_foot_car": "Minutos a pie / minutos en auto"},
            color_discrete_sequence=["#2b8cbe"],
        )
        fig.update_layout(height=360, yaxis_title="Centros poblados")
        st.plotly_chart(fig, **_ANCHO)


# ---------------------------------------------------------------------------
# Altitud
# ---------------------------------------------------------------------------
with tab_altitud:
    st.subheader("Tiempo de acceso frente a altitud")
    resumen_alt = M["altitud_resumen"]
    puntos_alt = M["altitud_puntos"].dropna(subset=["Z", "t_min"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlación de Spearman (ρ)", f"{float(resumen_alt['spearman_rho'].iloc[0]):.3f}")
    c2.metric("Valor p", f"{float(resumen_alt['p_value'].iloc[0]):.2e}")
    c3.metric("Puntos", f"{int(resumen_alt['n'].iloc[0]):,}")

    fig = px.scatter(
        puntos_alt, x="Z", y="t_min", opacity=0.35, log_y=True,
        labels={"Z": "Altitud (msnm)", "t_min": "Minutos al establecimiento resolutivo (log)"},
        color_discrete_sequence=["#08306b"],
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, **_ANCHO)

    st.warning(
        "**Correlación, no causalidad.** " + str(resumen_alt["nota_causalidad"].iloc[0]),
        icon="⚠️",
    )
    st.caption(
        "El signo de ρ además cambia de sentido según el filtro: con los tres departamentos, "
        "los peores tiempos están en la Amazonía **baja** (Loreto, <200 msnm), lo que empuja la "
        "correlación global hacia cero o la invierte. Filtra por Ayacucho para ver la relación "
        "dentro de un territorio andino homogéneo."
    )


# ---------------------------------------------------------------------------
# Calidad de datos
# ---------------------------------------------------------------------------
with tab_calidad:
    st.markdown(
        "Estas tablas son las que exige la consigna: **una fila por regla**, con cuántos registros "
        "se marcaron, qué se hizo y por qué. Ninguna regla descarta en silencio: cada fila marcada "
        "conserva su columna `flag_*` en `data/processed/`. Son las cifras del pipeline completo, "
        "no del filtro de la barra lateral."
    )

    st.subheader("Oferta — RENIPRESS (establecimientos de salud)")
    st.dataframe(dd.etiquetas_legibles(cargar_output("data_quality_report")),
                 hide_index=True, **_ANCHO)

    st.subheader("Demanda — centros poblados (SIGMED)")
    st.dataframe(dd.etiquetas_legibles(cargar_output("data_quality_report_demanda")),
                 hide_index=True, **_ANCHO)

    st.subheader("Enganche a la red vial (snapping, Fase 2)")
    st.dataframe(dd.etiquetas_legibles(cargar_output("snapping_report")),
                 hide_index=True, **_ANCHO)
    st.caption(
        "El 38 % de puntos de demanda sin red vial a menos de 1,000 m (perfil auto) es el dato más "
        "importante de esta tabla: no es un error del pipeline, es que en Loreto no hay carretera "
        "cerca del centro poblado. Los establecimientos, en cambio, enganchan al 100 %."
    )

    st.subheader("Cruce con población censada (INEI 2017)")
    st.dataframe(dd.etiquetas_legibles(cargar_output("poblacion_match_report")),
                 hide_index=True, **_ANCHO)

    st.subheader("Distancia de red vs. línea recta (calibración de Fase 2)")
    st.dataframe(cargar_output("calibracion_linea_recta"), hide_index=True, **_ANCHO)
    st.caption(
        "El factor de desvío mediano empírico es 1.59, por encima del 1.35 que traía `config.md` "
        "como valor por defecto. En Loreto la red desvía 1.87× **y** se recorre a 12.7 km/h "
        "medianos, frente a los 63 km/h de la sierra: el problema amazónico no es solo la forma "
        "de la red, es su velocidad."
    )

    st.subheader("Adquisición de fuentes (Fase 1)")
    adq = cargar_json("adquisicion")
    for fuente_nombre, info in adq.items():
        if info.get("status") == "ok":
            st.success(f"**{fuente_nombre}** — descargada correctamente.")
        else:
            st.error(f"**{fuente_nombre}** — no disponible automáticamente. {info.get('error', '')}")
    st.caption(
        "Que dos de las cuatro fuentes oficiales no se puedan descargar de forma programática es "
        "un hallazgo del trabajo sobre el estado de los datos abiertos peruanos, y está en el "
        "informe (sección Limitaciones), no escondido."
    )


# ---------------------------------------------------------------------------
# Metodología y descargas
# ---------------------------------------------------------------------------
with tab_metodo:
    st.subheader("Cómo se calcula lo que ves")
    st.markdown(
        """
- **Oferta.** RENIPRESS. "Resolutivo" = categoría II-1 o superior (`config.md > categorias_resolutivas`).
  Los establecimientos de categoría I (postas y centros de salud) **no** cuentan como destino en auto.
- **Demanda.** Centros poblados de SIGMED, con población del Directorio de Centros Poblados del
  Censo 2017 (INEI) empatada por código CCPP de 10 dígitos.
- **Tiempo de viaje.** OSRM sobre el extracto de OpenStreetMap de Perú, perfil `car`. Para cada
  centro poblado se toma el **mínimo** hacia cualquier establecimiento resolutivo.
- **Muestreo.** La matriz origen-destino se calculó sobre una muestra estratificada por distrito
  de 5,002 de los 19,460 centros poblados (`config.md > enrutamiento.muestreo`, semilla 42). Todos
  los agregados aplican el peso de diseño del muestreo además de la población censada.
- **Urbano/rural.** Proxy administrativo: es urbano el centro poblado que es capital de distrito,
  provincia o departamento. No es la definición del INEI (que usa densidad de manzanas censales,
  no disponible en este extracto).
- **Simulador.** `metrics.apply_upgrades` recalcula t_min como el mínimo entre el tiempo actual y
  el tiempo al candidato ascendido más cercano. Los tiempos hacia candidatos son estimados a
  partir de la calibración línea-recta/red mientras no exista la matriz OSRM correspondiente.
- **Recálculo con filtros.** Cada filtro reejecuta `src/metrics.py` sobre el subconjunto. Con todo
  seleccionado, los números de este panel son idénticos a los CSV de `data/outputs/`.
        """
    )

    st.subheader("Trazabilidad de la corrida")
    resumen_fase4 = cargar_json("fase4")
    if resumen_fase4:
        st.json(resumen_fase4, expanded=False)
    else:
        st.info("No hay `data/outputs/resumen_fase4.json`; corre `python run_fase4.py`.")

    st.subheader("Descargar los resultados")
    st.caption(
        "Los mismos archivos que cita el informe, sin renombrar columnas: lo que se descarga aquí "
        "es exactamente lo que produjo el pipeline."
    )
    descargables = [
        ("coverage_bands", "Población por banda de acceso"),
        ("weighted_mean_access_distrito", "Tiempo medio ponderado por distrito"),
        ("weighted_mean_access_provincia", "Tiempo medio ponderado por provincia"),
        ("weighted_mean_access_departamento", "Tiempo medio ponderado por departamento"),
        ("critical_gap_ranking", "Ranking de brechas críticas"),
        ("gini_access_resumen", "Gini del acceso"),
        ("urban_rural_contrast", "Contraste urbano / rural"),
        ("access_vs_altitude_resumen", "Acceso vs. altitud"),
        ("comparacion_modos", "Comparación de los tres modos"),
        ("comparacion_pie_vs_coche", "A pie vs. auto (puntos urbanos)"),
        ("calibracion_linea_recta", "Calibración línea recta vs. red"),
        ("data_quality_report", "Reporte de calidad — oferta"),
        ("data_quality_report_demanda", "Reporte de calidad — demanda"),
        ("snapping_report", "Reporte de enganche a la red vial"),
        ("poblacion_match_report", "Reporte del cruce con población censada"),
    ]
    cols = st.columns(2)
    for i, (nombre_csv, etiqueta) in enumerate(descargables):
        try:
            csv = cargar_output(nombre_csv).to_csv(index=False).encode("utf-8")
        except FileNotFoundError:
            continue
        cols[i % 2].download_button(
            f"⬇️ {etiqueta}", csv, file_name=f"{nombre_csv}.csv", mime="text/csv",
            key=f"dl_{nombre_csv}", **_ANCHO,
        )
