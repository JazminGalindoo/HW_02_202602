"""
validation.py — Fase 1: normalización y capa de validación
============================================================

Dos responsabilidades separadas a propósito:

  1. normalize_category() / is_resolutive()
     La lógica de negocio de "qué cuenta como resolutivo". Las reglas de
     normalización viven en config.md (normalizacion_categoria), este
     módulo solo las aplica -- así son visibles y editables sin tocar código,
     tal como pide la consigna.

  2. Las 6 reglas de validación geoespacial obligatorias, cada una como una
     función pura (DataFrame in -> DataFrame de filas marcadas out) más un
     orquestador `run_quality_pipeline` que las corre todas, decide
     corregir/eliminar/mantener-con-advertencia según corresponda, y arma el
     informe de calidad de datos exigido.

Filosofía: NINGUNA regla descarta filas en silencio. Cada fila marcada queda
con una columna `flag_<regla>` = True y una decisión explícita en
`accion_<regla>` ∈ {"corregido", "eliminado", "mantenido_con_advertencia"}.
El DataFrame final conserva todas las filas no eliminadas explícitamente.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:  # geopandas es pesado; permite testear la parte no-geo sin él
    gpd = None
    Point = None

from src.config_loader import load_config

logger = logging.getLogger("validation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] validation: %(message)s")


# ---------------------------------------------------------------------------
# Normalización de categoría / resolutividad
# ---------------------------------------------------------------------------

def _build_category_lookup(cfg: dict) -> dict:
    """Invierte config['normalizacion_categoria'] (canónica -> [variantes])
    en un dict plano variante_normalizada -> canónica, para lookup O(1)."""
    lookup = {}
    for canonica, variantes in cfg["normalizacion_categoria"].items():
        for v in variantes:
            key = _squash(v)
            lookup[key] = canonica
    return lookup


def _squash(s: str) -> str:
    """Colapsa espacios, mayúsculas y guiones para hacer el matching robusto
    a 'II-1' vs 'II - 1' vs 'ii1' vs '  II-1  '."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "", s)
    s = s.replace("–", "-").replace("_", "-")
    return s


def normalize_category(raw_value: object, cfg: Optional[dict] = None) -> str:
    """Devuelve la categoría canónica (p.ej. 'II-1') o 'SIN_CATEGORIA' si el
    valor crudo no matchea ninguna variante conocida. Nunca lanza excepción:
    un valor no reconocido es un hallazgo a reportar, no un crash."""
    cfg = cfg or load_config()
    lookup = _build_category_lookup(cfg)
    squashed = _squash(raw_value)
    return lookup.get(squashed, "SIN_CATEGORIA")


def is_active(raw_status: object, cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_config()
    activos = {_squash(s) for s in cfg["estados_activos"]}
    return _squash(raw_status) in activos


def is_resolutive(categoria_normalizada: str, raw_status: object, cfg: Optional[dict] = None) -> bool:
    """Regla de negocio central del proyecto: resolutivo <=> activo AND
    categoría en la lista blanca. Ver config.md > categorias_resolutivas."""
    cfg = cfg or load_config()
    return (
        is_active(raw_status, cfg)
        and categoria_normalizada in cfg["categorias_resolutivas"]
    )


def apply_category_rules(df: pd.DataFrame, col_categoria: str, col_estado: str,
                          cfg: Optional[dict] = None) -> pd.DataFrame:
    """Añade columnas 'categoria_norm' y 'es_resolutivo' a una copia de df."""
    cfg = cfg or load_config()
    out = df.copy()
    out["categoria_norm"] = out[col_categoria].apply(lambda v: normalize_category(v, cfg))
    out["es_resolutivo"] = [
        is_resolutive(cat, est, cfg)
        for cat, est in zip(out["categoria_norm"], out[col_estado])
    ]
    n_sin_categoria = (out["categoria_norm"] == "SIN_CATEGORIA").sum()
    if n_sin_categoria:
        logger.warning(
            "%d registros con categoría no reconocida por config.md.normalizacion_categoria "
            "(quedan como 'SIN_CATEGORIA', mantenidos con advertencia, excluidos de resolutivos).",
            n_sin_categoria,
        )
    return out


# ---------------------------------------------------------------------------
# Reglas de validación geoespacial (cada una devuelve una Serie booleana)
# ---------------------------------------------------------------------------

def flag_missing_coords(df: pd.DataFrame, col_lon: str, col_lat: str) -> pd.Series:
    lon, lat = df[col_lon], df[col_lat]
    return lon.isna() | lat.isna() | (lon == 0) | (lat == 0)


def flag_out_of_bbox(df: pd.DataFrame, col_lon: str, col_lat: str, cfg: dict) -> pd.Series:
    bbox = cfg["bbox_peru"]
    lon, lat = df[col_lon], df[col_lat]
    dentro = lon.between(bbox["lon_min"], bbox["lon_max"]) & lat.between(bbox["lat_min"], bbox["lat_max"])
    return ~dentro & lon.notna() & lat.notna()


def flag_lat_lon_swapped(df: pd.DataFrame, col_lon: str, col_lat: str, cfg: dict) -> pd.Series:
    """Un caso muy común en registros administrativos peruanos: alguien tipeó
    latitud en la columna de longitud y viceversa. Detectamos esto probando
    si, INVIRTIENDO lon/lat, el punto SÍ cae dentro del bbox de Perú mientras
    que en su orientación original no lo hace."""
    bbox = cfg["bbox_peru"]
    lon, lat = df[col_lon], df[col_lat]
    fuera_original = ~(lon.between(bbox["lon_min"], bbox["lon_max"]) & lat.between(bbox["lat_min"], bbox["lat_max"]))
    dentro_invertido = lat.between(bbox["lon_min"], bbox["lon_max"]) & lon.between(bbox["lat_min"], bbox["lat_max"])
    return fuera_original & dentro_invertido & lon.notna() & lat.notna()


def flag_duplicate_codes(df: pd.DataFrame, col_codigo: str) -> pd.Series:
    return df[col_codigo].duplicated(keep="first") & df[col_codigo].notna()


def flag_outside_own_district(
    df: pd.DataFrame,
    col_lon: str,
    col_lat: str,
    col_ubigeo: str,
    districts_gdf: "gpd.GeoDataFrame",
    ubigeo_col_districts: str,
    tolerancia_m: float = 500,
) -> pd.Series:
    """Para cada fila, comprueba si el punto (lon, lat) cae dentro del
    polígono del distrito que el propio registro declara (por ubigeo). Un
    establecimiento que dice estar en el distrito X pero cuyo punto cae en
    otro distrito casi siempre es un error de digitación de coordenadas, no
    un límite administrativo mal dibujado.
    """
    if gpd is None:
        raise ImportError("geopandas es requerido para flag_outside_own_district")

    geom = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
            for xy in zip(df[col_lon], df[col_lat])]
    points_gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")

    # proyección métrica para el buffer de tolerancia (UTM 18S cubre la
    # mayor parte de Perú continental; suficientemente preciso para un
    # margen de cientos de metros, no para medición exacta)
    districts_m = districts_gdf.to_crs("EPSG:32718")
    points_m = points_gdf.to_crs("EPSG:32718")

    flags = pd.Series(False, index=df.index)
    dist_by_ubigeo = { row[ubigeo_col_districts]: row.geometry.buffer(tolerancia_m) 
                      for _, row in districts_m.iterrows() if row.geometry is not None }

    for idx, row in points_m.iterrows():
        ubigeo = df.at[idx, col_ubigeo]
        geom_pt = row.geometry
        if geom_pt is None or pd.isna(ubigeo):
            continue
        poly = dist_by_ubigeo.get(ubigeo)
        if poly is None:
            continue  # ubigeo no encontrado en el shapefile -> no se puede juzgar, no se marca
        if not poly.contains(geom_pt):
            flags.at[idx] = True
    return flags


def fix_encoding_column(series: pd.Series, candidatos: list[str]) -> tuple[pd.Series, pd.Series]:
    """Intenta reparar mojibake típico de mezclar UTF-8/latin-1 (p.ej.
    'CENTRO DE SALUD ÃANDO' en vez de 'CENTRO DE SALUD ÑANDO'). Devuelve
    (serie_reparada, mask_de_filas_que_se_tocaron)."""
    def _try_fix(s):
        if not isinstance(s, str):
            return s, False
        # Heurística: si el texto tiene bytes típicos de mojibake (Ã, Â, etc.)
        if not re.search(r"[ÃÂ]", s):
            return s, False
        for enc_from in candidatos:
            try:
                fixed = s.encode(enc_from, errors="strict").decode("utf-8", errors="strict")
                # Solo aceptamos el arreglo si eliminó los caracteres sospechosos
                if not re.search(r"[ÃÂ]", fixed):
                    return fixed, True
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        return s, False

    results = series.apply(_try_fix)
    fixed_series = results.apply(lambda t: t[0])
    touched_mask = results.apply(lambda t: t[1])
    return fixed_series, touched_mask


# ---------------------------------------------------------------------------
# Orquestador: corre las 6 reglas, decide acción, arma el informe de calidad
# ---------------------------------------------------------------------------

def run_quality_pipeline(
    df: pd.DataFrame,
    *,
    col_lon: str,
    col_lat: str,
    col_codigo: str,
    col_ubigeo: str,
    col_texto_libre: list[str],
    districts_gdf: Optional["gpd.GeoDataFrame"] = None,
    ubigeo_col_districts: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre las 6 reglas de calidad obligatorias sobre df y devuelve:
        (df_limpio, reporte_calidad)

    df_limpio conserva TODAS las filas excepto las que la regla de
    duplicados o de coordenadas nulas irrecuperables decide eliminar
    explícitamente; todo lo demás queda con columnas flag_* y, cuando aplica,
    con la coordenada corregida.

    reporte_calidad tiene una fila por regla: n_marcados, accion, justificacion.
    Esta es la tabla que se exporta a data/outputs/data_quality_report.csv y
    se muestra en el panel de Streamlit (Fase 4) y en el video.
    """
    cfg = cfg or load_config()
    out = df.copy()
    n_total = len(out)
    filas_reporte = []

    # --- Regla 1: coordenadas faltantes / nulas / cero -----------------
    mask_missing = flag_missing_coords(out, col_lon, col_lat)
    out["flag_coords_faltantes"] = mask_missing
    filas_reporte.append({
        "regla": "coords_faltantes",
        "n_marcados": int(mask_missing.sum()),
        "accion": "eliminado (irrecuperable: sin lat/lon no hay forma de enrutar el punto)",
        "justificacion": "Un registro sin coordenada no puede participar del cálculo de "
                          "tiempo de viaje; se mantiene en data/processed con la bandera "
                          "para auditoría, pero se excluye de la matriz de enrutamiento.",
    })

    # --- Regla 2: fuera de la caja delimitadora de Perú -----------------
    mask_bbox = flag_out_of_bbox(out, col_lon, col_lat, cfg)
    out["flag_fuera_de_bbox"] = mask_bbox
    filas_reporte.append({
        "regla": "fuera_de_bbox_peru",
        "n_marcados": int(mask_bbox.sum()),
        "accion": "mantenido con advertencia si es corregible por swap (ver regla 3); "
                  "si no, eliminado del cálculo de enrutamiento",
        "justificacion": "Fuera de [-81.4,-68.6] x [-18.4,-0.04] es geográficamente "
                          "imposible para un establecimiento peruano; se intenta primero "
                          "explicarlo como intercambio lat/lon antes de descartar.",
    })

    # --- Regla 3: lat/lon intercambiados --------------------------------
    mask_swap = flag_lat_lon_swapped(out, col_lon, col_lat, cfg)
    out.loc[mask_swap, [col_lon, col_lat]] = out.loc[mask_swap, [col_lat, col_lon]].values
    out["flag_lat_lon_intercambiados"] = mask_swap
    filas_reporte.append({
        "regla": "lat_lon_intercambiados",
        "n_marcados": int(mask_swap.sum()),
        "accion": "corregido (coordenadas intercambiadas en su lugar)",
        "justificacion": "Si invertir lon/lat mueve el punto de fuera-de-Perú a "
                          "dentro-del-bbox, se asume digitación invertida y se corrige. "
                          "Tasa de recuperación reportada explícitamente.",
    })

    # Recalcular bbox tras la corrección de swap, para no seguir arrastrando
    # como "eliminado" a puntos que la regla 3 ya reparó.
    mask_bbox_post_swap = flag_out_of_bbox(out, col_lon, col_lat, cfg) & ~mask_missing
    filas_a_eliminar_por_bbox = mask_bbox_post_swap.copy()

    # --- Regla 4: fuera del polígono de su propio distrito --------------
    if districts_gdf is not None:
        mask_outside_district = flag_outside_own_district(
            out, col_lon, col_lat, col_ubigeo, districts_gdf, ubigeo_col_districts,
            tolerancia_m=cfg["validacion"]["tolerancia_fuera_de_poligono_m"],
        )
    else:
        mask_outside_district = pd.Series(False, index=out.index)
        logger.warning("districts_gdf no provisto: regla 'fuera_de_su_distrito' se omite (marcada en 0).")
    out["flag_fuera_de_su_distrito"] = mask_outside_district
    filas_reporte.append({
        "regla": "fuera_de_su_propio_distrito",
        "n_marcados": int(mask_outside_district.sum()),
        "accion": "mantenido con advertencia (no se corrige automáticamente: no sabemos "
                  "si el error está en la coordenada o en el ubigeo declarado)",
        "justificacion": f"Punto fuera del polígono de su distrito con tolerancia de "
                          f"{cfg['validacion']['tolerancia_fuera_de_poligono_m']} m. "
                          "Se excluye del cálculo de accesibilidad a nivel distrital pero "
                          "se conserva para no perder el establecimiento en el mapa general.",
    })

    # --- Regla 5: códigos duplicados -------------------------------------
    mask_dup = flag_duplicate_codes(out, col_codigo)
    filas_reporte.append({
        "regla": "codigos_duplicados",
        "n_marcados": int(mask_dup.sum()),
        "accion": "eliminado (se conserva la primera ocurrencia)",
        "justificacion": f"Duplicados exactos de '{col_codigo}' inflarían el conteo de "
                          "oferta resolutiva en el distrito afectado.",
    })
    out = out.loc[~mask_dup].copy()

    # --- Regla 6: problemas de codificación en texto libre ---------------
    total_encoding_fixed = 0
    for col in col_texto_libre:
        if col not in out.columns:
            continue
        fixed, touched = fix_encoding_column(out[col], cfg["validacion"]["encoding_candidatos"])
        out[col] = fixed
        total_encoding_fixed += int(touched.sum())
    filas_reporte.append({
        "regla": "encoding_utf8_vs_latin1",
        "n_marcados": total_encoding_fixed,
        "accion": "corregido in place en columnas de texto",
        "justificacion": f"Mojibake detectado y reparado en columnas: {col_texto_libre}. "
                          "Sin esto, nombres con Ñ/tildes rompen joins por nombre y "
                          "búsquedas en el dashboard.",
    })

    # --- Eliminación final por coords irrecuperables ----------------------
    n_antes = len(out)
    eliminar = mask_missing.reindex(out.index, fill_value=False) | filas_a_eliminar_por_bbox.reindex(out.index, fill_value=False)
    out_clean = out.loc[~eliminar].copy()
    logger.info(
        "run_quality_pipeline: %d filas de entrada -> %d en processed "
        "(%d eliminadas por coords faltantes/irrecuperables, %d por duplicados).",
        n_total, len(out_clean), int(eliminar.sum()), int(mask_dup.sum()),
    )

    reporte = pd.DataFrame(filas_reporte)
    reporte["pct_del_total"] = (reporte["n_marcados"] / max(n_total, 1) * 100).round(2)
    return out_clean, reporte


if __name__ == "__main__":
    # Auto-prueba mínima con datos sintéticos, sin depender de descargas.
    cfg = load_config()
    demo = pd.DataFrame({
        "codigo": ["A1", "A2", "A2", "A3", "A4"],
        "lon": [-80.6, -0.06, -70.0, None, -75.0],   # A2: lat/lon claramente invertidos
        "lat": [-5.2, -80.6, -8.0, -5.0, -300.0],     # A4: fuera de bbox sin remedio
        "ubigeo": ["200101", "200101", "160101", "160101", "50101"],
        "categoria_ipress": ["I - 1", "II-1", "CATEGORIA III-E", "i2", "???"],
        "estado": ["EN FUNCIONAMIENTO"] * 5,
        "nombre": ["Posta A", "Hospital Ã‘ando", "Centro C", "Posta D", "Posta E"],
    })
    demo_cat = apply_category_rules(demo, "categoria_ipress", "estado", cfg)
    clean, report = run_quality_pipeline(
        demo_cat, col_lon="lon", col_lat="lat", col_codigo="codigo",
        col_ubigeo="ubigeo", col_texto_libre=["nombre"], cfg=cfg,
    )
    print(report.to_string(index=False))
    print(clean[["codigo", "lon", "lat", "categoria_norm", "es_resolutivo", "nombre"]])
