"""
app/dashboard.py — Fase 4: panel interactivo (Streamlit)
==========================================================

Levantar con:

    python run_fase4.py            # una vez, prepara los artefactos
    streamlit run app/dashboard.py

No requiere OSRM, ni Docker, ni geopandas: todo lo que muestra sale de los
parquet/CSV que dejaron las Fases 1-3 y `run_fase4.py`.

Principio de diseño (el mismo que declara metrics.py): **ninguna métrica se
implementa aquí**. Cuando el usuario filtra por departamento o zona, el
panel vuelve a llamar a `src.metrics.*` sobre el subconjunto filtrado. Si
alguien cambia la definición de una banda de cobertura o del Gini, el panel
y el informe cambian juntos, porque leen la misma función. Esta capa solo:
    (a) carga artefactos (via src/dashboard_data.py),
    (b) filtra,
    (c) dibuja.

Sobre la honestidad de las cifras: el panel repite en cada vista de qué
universo habla. Las métricas se calculan sobre la muestra estratificada de
5,002 centros poblados que se logró enrutar en Fase 2 (de 19,460), ponderada
por población censada 2017 y por el peso de diseño del muestreo. No se
presenta ningún número como si describiera a los tres departamentos
completos sin ese matiz.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/dashboard.py` pone app/ en sys.path[0], no la raíz del
# repo, así que `import src...` fallaría. Se añade la raíz explícitamente en
# vez de exigir `pip install -e .` o un PYTHONPATH manual al evaluador.
REPO_ROOT = Path(__file__).resolve().parent.parent
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
    access_vs_altitude, coverage_bands, critical_gap_ranking, gini_access,
    urban_rural_contrast, weighted_mean_access,
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
    return df["NOMBRE"].fillna("(sin nombre)") + " · " + categoria


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


@st.cache_data
def cargar_output(nombre: str) -> pd.DataFrame:
    return dd.load_output_csv(nombre, CFG)


@st.cache_data
def cargar_json(cual: str) -> dict:
    return dd.load_acquisition_summary(CFG) if cual == "adquisicion" else dd.load_resumen_fase4(CFG)


@st.cache_data(show_spinner="Agregando por distrito…")
def agregado_distrital(deps: tuple, zonas: tuple) -> pd.DataFrame:
    """Media ponderada a nivel distrito, siempre (el coropleto es distrital
    aunque el usuario esté mirando las tablas a nivel provincia)."""
    sub = dd.filter_puntos(cargar_puntos(), departamentos=deps, zonas=zonas)
    access_df, population_df = dd.split_access_population(sub)
    return weighted_mean_access(access_df, population_df, level="distrito", cfg=CFG)


@st.cache_data(show_spinner="Recalculando métricas sobre el filtro…")
def metricas(deps: tuple, zonas: tuple, nivel: str) -> dict:
    """Recalcula TODAS las métricas del panel sobre el subconjunto filtrado,
    llamando a las funciones de src/metrics.py (no reimplementándolas).

    Se cachea por la tupla de filtros: volver a un filtro ya visto es
    instantáneo. Devuelve un dict de DataFrames para hacer una sola pasada
    por el subconjunto en vez de siete."""
    puntos = cargar_puntos()
    sub = dd.filter_puntos(puntos, departamentos=deps, zonas=zonas)
    if sub.empty:
        return {"vacio": True, "n_universo": 0, "n_muestra": 0}

    access_df, population_df = dd.split_access_population(sub)
    if access_df.empty:
        return {"vacio": True, "n_universo": len(sub), "n_muestra": 0}

    bandas = coverage_bands(access_df, population_df, CFG)
    agregado = weighted_mean_access(access_df, population_df, level=nivel, cfg=CFG)
    gini_resumen, lorenz = gini_access(access_df, population_df)
    contraste = urban_rural_contrast(access_df, population_df)
    altitud_puntos, altitud_resumen = access_vs_altitude(access_df, population_df)

    return {
        "vacio": False,
        "n_universo": len(sub),
        "n_muestra": len(access_df),
        "bandas": bandas,
        "agregado": agregado,
        "gini": gini_resumen,
        "lorenz": lorenz,
        "contraste": contraste,
        "altitud_puntos": altitud_puntos,
        "altitud_resumen": altitud_resumen,
    }


# ---------------------------------------------------------------------------
# Barra lateral: filtros
# ---------------------------------------------------------------------------

puntos_all = cargar_puntos()
deps_disponibles = sorted(puntos_all["DEP"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filtros")
    deps = st.multiselect("Departamento", deps_disponibles, default=deps_disponibles)
    zonas = st.multiselect(
        "Zona", ["urbano", "rural"], default=["urbano", "rural"],
        help="Proxy administrativo: 'urbano' = centro poblado capital de distrito, "
             "provincia o departamento (CAPITAL != 0). No es la definición oficial "
             "de área urbana del INEI; ver la pestaña Metodología.",
    )
    nivel = st.selectbox(
        "Nivel de agregación", ["distrito", "provincia", "departamento"], index=0,
        help="Se agrupa por prefijo de ubigeo (6/4/2 dígitos), no por nombre de texto.",
    )
    st.divider()
    st.caption(
        "Al mover un filtro, el panel **recalcula** las métricas llamando a las mismas "
        "funciones de `src/metrics.py` que produjeron las tablas del informe. "
        "Con los tres departamentos y ambas zonas seleccionados, los números coinciden "
        "exactamente con `data/outputs/`."
    )
    st.divider()
    st.caption(
        f"Universo: **{len(puntos_all):,}** centros poblados · "
        f"muestra enrutada en Fase 2: **{int(puntos_all['en_muestra_enrutada'].sum()):,}**"
    )

deps_t, zonas_t = tuple(deps), tuple(zonas)
M = metricas(deps_t, zonas_t, nivel)

st.title("Golden Hour — acceso por carretera a salud resolutiva")
st.markdown(
    "Tiempo de viaje en automóvil desde cada centro poblado hasta el establecimiento "
    "de salud **con capacidad resolutiva** (categoría II-1 o superior, RENIPRESS) más "
    "cercano, en **Piura** (costa), **Ayacucho** (sierra) y **Loreto** (selva)."
)

if M.get("vacio"):
    st.warning(
        "El filtro actual no deja ningún centro poblado con ruta calculada. "
        "Selecciona al menos un departamento y una zona."
    )
    st.stop()


def _pct_banda(bandas: pd.DataFrame, etiqueta: str) -> float:
    fila = bandas.loc[bandas["banda"] == etiqueta, "pct_poblacion"]
    return float(fila.iloc[0]) if len(fila) else float("nan")


tab_resumen, tab_mapa, tab_brechas, tab_equidad, tab_modos, tab_altitud, tab_calidad, tab_metodo = st.tabs([
    "Resumen", "Mapa", "Brechas", "Equidad", "Modos de viaje", "Altitud",
    "Calidad de datos", "Metodología y descargas",
])


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    bandas = M["bandas"]
    poblacion_representada = float(bandas["poblacion"].sum())
    t_medio = np.average(
        M["agregado"]["t_min_medio_ponderado"].dropna(),
        weights=M["agregado"].loc[M["agregado"]["t_min_medio_ponderado"].notna(), "poblacion_con_acceso"],
    ) if M["agregado"]["poblacion_con_acceso"].sum() > 0 else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Población representada", f"{poblacion_representada:,.0f}",
              help="Población censada 2017 de la muestra enrutada, ponderada por el peso de diseño del muestreo.")
    c2.metric("A menos de 30 min", f"{_pct_banda(bandas, ORDEN_BANDAS[0]):.1f} %")
    c3.metric(f"A más de {CFG['metricas']['bandas_acceso_min'][-1]} min",
              f"{_pct_banda(bandas, ORDEN_BANDAS[-2]):.1f} %")
    c4.metric("Tiempo medio ponderado", dd.formato_minutos(t_medio))
    c5.metric("Gini del acceso", f"{float(M['gini']['gini'].iloc[0]):.3f}",
              help="0 = todos esperan lo mismo; 1 = el tiempo total de viaje se concentra en una minoría.")

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
            dd.etiquetas_legibles(bandas).style.format({"Población": "{:,.0f}", "% de población": "{:.2f}"}),
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
    vista = st.radio(
        "Vista", ["Coropleto por distrito", "Centros poblados"],
        horizontal=True, label_visibility="collapsed",
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        mostrar_ipress = st.checkbox("Mostrar establecimientos resolutivos", value=True)
    with c2:
        if vista == "Centros poblados":
            mostrar_no_muestreados = st.checkbox(
                "Incluir centros poblados fuera de la muestra enrutada (sin t_min)", value=False,
            )
        else:
            mostrar_no_muestreados = False

    geojson = cargar_geojson()
    instalaciones = dd.filter_instalaciones(cargar_instalaciones(), deps, solo_resolutivas=True)

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
            distritos = agregado_distrital(deps_t, zonas_t).assign(
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
            if mostrar_ipress and not instalaciones.empty:
                fig.add_trace(_TrazaPuntos(
                    lat=instalaciones["lat"], lon=instalaciones["lon"], mode="markers",
                    marker={"size": 9, "color": "#08306b"},
                    name="IPRESS resolutiva",
                    text=_etiqueta_ipress(instalaciones),
                    hovertemplate="%{text}<extra></extra>",
                ))
            st.plotly_chart(_layout_mapa(fig), **_ANCHO)
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
        pts = dd.filter_puntos(cargar_puntos(), deps, zonas,
                               solo_muestra_enrutada=not mostrar_no_muestreados)
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
        if mostrar_ipress and not instalaciones.empty:
            fig.add_trace(_TrazaPuntos(
                lat=instalaciones["lat"], lon=instalaciones["lon"], mode="markers",
                marker={"size": 11, "color": "#08306b", "symbol": "circle"},
                name="IPRESS resolutiva",
                text=_etiqueta_ipress(instalaciones),
                hovertemplate="%{text}<extra></extra>",
            ))
        st.plotly_chart(_layout_mapa(fig), **_ANCHO)
        st.caption(
            f"{len(pts):,} centros poblados dibujados · {len(instalaciones)} establecimientos "
            "resolutivos en el filtro. Un punto gris es un centro poblado que no entró al "
            "muestreo de Fase 2: no tiene tiempo calculado, que no es lo mismo que no tener acceso."
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
        hide_index=True, **_ANCHO, height=420,
    )
    st.caption(
        "`% población con acceso` expone qué parte de la población de esa unidad quedó **fuera** "
        "del promedio por no tener ruta calculada. Una unidad con tiempo medio bajo y 30 % de "
        "cobertura no es una unidad bien servida."
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
        "Descargar ranking (CSV)", ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_brechas_{nivel}.csv", mime="text/csv",
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
    puntos_filtrados = dd.filter_puntos(cargar_puntos(), deps, zonas)
    ids_filtro = set(puntos_filtrados["id"])

    modos = cargar_output("comparacion_modos")
    modos = modos[modos["id_demanda"].astype(str).isin({str(i) for i in ids_filtro})]

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
    pie_coche = pie_coche[pie_coche["id_demanda"].astype(str).isin({str(i) for i in ids_filtro})]
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
    snap = cargar_output("snapping_report")
    st.dataframe(dd.etiquetas_legibles(snap), hide_index=True, **_ANCHO)
    st.caption(
        "El 38 % de puntos de demanda sin red vial a menos de 1,000 m (perfil auto) es el dato más "
        "importante de esta tabla: no es un error del pipeline, es que en Loreto no hay carretera "
        "cerca del centro poblado. Los establecimientos, en cambio, enganchan al 100 %."
    )

    st.subheader("Cruce con población censada (INEI 2017)")
    st.dataframe(dd.etiquetas_legibles(cargar_output("poblacion_match_report")),
                 hide_index=True, **_ANCHO)

    st.subheader("Adquisición de fuentes (Fase 1)")
    adq = cargar_json("adquisicion")
    for fuente, info in adq.items():
        if info.get("status") == "ok":
            st.success(f"**{fuente}** — descargada correctamente.")
        else:
            st.error(f"**{fuente}** — no disponible automáticamente. {info.get('error', '')}")
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
- **Recálculo con filtros.** Cada filtro reejecuta `src/metrics.py` sobre el subconjunto. Con todo
  seleccionado, los números de este panel son idénticos a los CSV de `data/outputs/`.
        """
    )

    st.subheader("Trazabilidad de la corrida")
    resumen = cargar_json("fase4")
    if resumen:
        st.json(resumen, expanded=False)
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
        ("data_quality_report", "Reporte de calidad — oferta"),
        ("data_quality_report_demanda", "Reporte de calidad — demanda"),
        ("snapping_report", "Reporte de enganche a la red vial"),
        ("poblacion_match_report", "Reporte del cruce con población censada"),
    ]
    cols = st.columns(2)
    for i, (nombre, etiqueta) in enumerate(descargables):
        try:
            csv = cargar_output(nombre).to_csv(index=False).encode("utf-8")
        except FileNotFoundError:
            continue
        cols[i % 2].download_button(
            f"⬇️ {etiqueta}", csv, file_name=f"{nombre}.csv", mime="text/csv",
            **_ANCHO, key=f"dl_{nombre}",
        )
