"""
run_fase4.py — Fase 4 (preparación): artefactos que consume el panel
=====================================================================

El panel de Streamlit (`app/dashboard.py`) tiene que abrir en segundos, sin
OSRM levantado, sin Docker y sin geopandas instalado. Para eso este script
deja pre-cocinados tres artefactos y un resumen de trazabilidad:

    data/processed/puntos_dashboard.parquet
        1 fila por centro poblado del universo (~19,460), con: geografía,
        población censada 2017, zona urbano/rural, peso de muestreo de
        Fase 2 y -- para los que entraron a la muestra enrutada -- t_min
        hacia la IPRESS resolutiva más cercana y su banda de acceso.

    data/processed/instalaciones_dashboard.parquet
        1 fila por IPRESS con coordenada válida (oferta), con la categoría
        normalizada y la bandera es_resolutivo de Fase 1.

    data/processed/distritos_3dep.geojson
        Los 226 polígonos distritales de Piura, Ayacucho y Loreto,
        recortados del GeoJSON nacional y con las coordenadas redondeadas,
        para que el coropleto no arrastre 1.8 MB de polígonos de los otros
        22 departamentos.

    data/outputs/resumen_fase4.json
        Qué se generó, cuándo y a partir de qué (pestaña de reproducibilidad).

Por qué materializar esto y no calcularlo dentro del panel: Streamlit
re-ejecuta el script entero en cada interacción del usuario. Reproducir el
muestreo estratificado de Fase 2 y recorrer la matriz OD de 290,116 filas en
cada clic haría el panel inusable; y peor, mezclaría la construcción del
dato con su presentación. Aquí se construye una vez; el panel solo lee y
filtra.

Uso:
    python run_fase4.py
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from src.config_loader import REPO_ROOT, load_config, path_for
from src.dashboard_data import (
    DISTRITOS_GEOJSON, INSTALACIONES_DASHBOARD, PUNTOS_DASHBOARD, RESUMEN_FASE4,
    assign_bands,
)
from src.metrics import access_time, attach_sample_weights, classify_urban_rural

logger = logging.getLogger("fase4")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Handler propio en vez de logging.basicConfig(): al importar src.metrics
    # se importa src.routing, que ya llamó a basicConfig() con 'routing'
    # hardcodeado en el formato, y basicConfig() es un no-op en llamadas
    # posteriores -- los mensajes de este script saldrían firmados 'routing'.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] fase4: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# Precisión del redondeo de coordenadas del GeoJSON recortado. 5 decimales
# ~ 1 m en el ecuador: sobra para un coropleto distrital y recorta el
# archivo a menos de la mitad. No se usa simplificación topológica (Douglas-
# Peucker) porque exigiría shapely/geopandas en la ruta del panel y podría
# abrir huecos entre distritos vecinos.
DECIMALES_GEOJSON = 5

# Propiedades que sobreviven al recorte. El GeoJSON de origen trae 17
# (OBJECTID, SHAPE_LENG, AREA_MINAM, ...) que el panel no usa.
PROPS_GEOJSON = ["IDDIST", "IDPROV", "IDDPTO", "NOMBDIST", "NOMBPROV", "NOMBDEP"]


def _redondear_coords(obj, nd: int = DECIMALES_GEOJSON):
    """Redondea recursivamente las coordenadas de una geometría GeoJSON
    (listas anidadas a profundidad arbitraria: Polygon vs. MultiPolygon)."""
    if isinstance(obj, (int, float)):
        return round(float(obj), nd)
    return [_redondear_coords(x, nd) for x in obj]


def construir_geojson_distritos(cfg: dict) -> Path | None:
    """Recorta el GeoJSON distrital nacional a los 3 departamentos.

    Si el archivo crudo no está (está en .gitignore: se regenera con
    `python -m src.acquisition`), intenta descargarlo una vez. Si tampoco se
    puede -- portal caído, sin red en la máquina del evaluador --, avisa y
    devuelve None: el panel degrada al mapa de puntos en vez de reventar.
    Es el mismo criterio de Fase 1: fallar documentando, no en silencio."""
    crudo = path_for("limites_distritales_raw", cfg)
    if not crudo.exists():
        logger.warning("No hay %s; intentando descargarlo (src.acquisition).", crudo.name)
        try:
            from src.acquisition import download_admin_boundaries
            crudo = download_admin_boundaries(cfg)
        except Exception as exc:  # noqa: BLE001 -- se documenta y se sigue
            logger.error(
                "No se pudo obtener el GeoJSON distrital (%s). El panel funcionará "
                "sin coropleto. Para habilitarlo: python -m src.acquisition", exc,
            )
            return None

    ubigeos_dep = {d["ubigeo_dep"] for d in cfg["departamentos"].values()}
    with crudo.open(encoding="utf-8") as fh:
        nacional = json.load(fh)

    features = []
    sin_geometria = []
    for f in nacional["features"]:
        iddist = str(f["properties"].get("IDDIST", ""))
        if iddist[:2] not in ubigeos_dep:
            continue
        if not f.get("geometry"):
            # El GeoJSON nacional (INEI 2007 georreferenciado por terceros)
            # trae algún distrito con geometry null. Se registra el ubigeo
            # en vez de descartarlo callado: ese distrito simplemente no se
            # pintará en el coropleto y hay que poder decir cuál es.
            sin_geometria.append(iddist)
            continue
        features.append({
            "type": "Feature",
            "id": iddist,  # Plotly hace el join del coropleto por feature.id
            "properties": {k: f["properties"].get(k) for k in PROPS_GEOJSON},
            "geometry": {
                "type": f["geometry"]["type"],
                "coordinates": _redondear_coords(f["geometry"]["coordinates"]),
            },
        })

    dest = REPO_ROOT / DISTRITOS_GEOJSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, ensure_ascii=False)
    logger.info(
        "GeoJSON recortado: %d distritos de %s -> %s (%.2f MB, el nacional pesa %.2f MB).",
        len(features), sorted(ubigeos_dep), DISTRITOS_GEOJSON,
        dest.stat().st_size / 1e6, crudo.stat().st_size / 1e6,
    )
    if sin_geometria:
        logger.warning(
            "%d distrito(s) de los 3 departamentos vienen con geometry null en la fuente y "
            "quedan fuera del coropleto: %s. Documentado en el informe (Limitaciones).",
            len(sin_geometria), sin_geometria,
        )
    return dest


def construir_puntos(cfg: dict) -> pd.DataFrame:
    """Une demanda + población + zona + peso muestral + t_min en una tabla.

    Reutiliza tal cual las funciones de Fase 3 (attach_sample_weights,
    classify_urban_rural, access_time): el panel no puede mostrar un t_min
    calculado de otra manera que el del informe."""
    demanda = pd.read_parquet(path_for("demanda_con_poblacion", cfg))
    ipress = pd.read_parquet(path_for("ipress_processed", cfg))
    matriz_car = pd.read_parquet(path_for("matriz_car", cfg))
    resolutivas_ids = set(ipress.loc[ipress["es_resolutivo"], "COD_IPRESS"])

    poblacion = classify_urban_rural(attach_sample_weights(demanda, cfg))
    acceso = access_time(matriz_car, resolutivas_ids)

    puntos = poblacion.merge(acceso, left_on="id", right_on="id_demanda", how="left")
    puntos["en_muestra_enrutada"] = puntos["id_demanda"].notna()
    # flag_sin_acceso_enrutable solo tiene sentido dentro de la muestra: un
    # punto que nunca se enrutó no es "sin acceso", es "no observado".
    puntos["flag_sin_acceso_enrutable"] = (
        puntos["flag_sin_acceso_enrutable"].fillna(False).astype(bool) & puntos["en_muestra_enrutada"]
    )
    puntos["banda"] = assign_bands(puntos["t_min"], cfg)
    puntos.loc[~puntos["en_muestra_enrutada"], "banda"] = "no_muestreado"

    # 'geometry' viene como WKB en bytes desde Fase 1: el panel usa lon/lat
    # y arrastrarla obligaría a geopandas para nada. 'id_demanda' es
    # redundante con 'id' tras el merge.
    puntos = puntos.drop(columns=[c for c in ["geometry", "id_demanda"] if c in puntos.columns])

    logger.info(
        "puntos_dashboard: %d centros poblados; %d en la muestra enrutada; "
        "%d con población censada conocida; %d urbanos.",
        len(puntos), int(puntos["en_muestra_enrutada"].sum()),
        int(puntos["poblacion_censada"].notna().sum()), int((puntos["zona"] == "urbano").sum()),
    )
    return puntos


def construir_instalaciones(cfg: dict) -> pd.DataFrame:
    """Capa de oferta del mapa. NORTE/ESTE de RENIPRESS son, pese al nombre,
    latitud y longitud en grados decimales (verificado en Fase 1); se
    renombran a lat/lon para que el panel no tenga que recordarlo."""
    ipress = pd.read_parquet(path_for("ipress_processed", cfg))
    cols = [
        "COD_IPRESS", "NOMBRE", "INSTITUCION", "CATEGORIA", "categoria_norm", "es_resolutivo",
        "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "UBIGEO", "NORTE", "ESTE",
        "flag_fuera_de_su_distrito",
    ]
    out = ipress[[c for c in cols if c in ipress.columns]].rename(columns={"NORTE": "lat", "ESTE": "lon"})
    logger.info(
        "instalaciones_dashboard: %d IPRESS con coordenada válida, de las cuales %d resolutivas "
        "(categoría %s) y %d sin categoría legible en el registro.",
        len(out), int(out["es_resolutivo"].sum()), "/".join(cfg["categorias_resolutivas"]),
        int((out["categoria_norm"] == "SIN_CATEGORIA").sum()),
    )
    return out


def main() -> None:
    cfg = load_config()

    puntos = construir_puntos(cfg)
    dest_puntos = REPO_ROOT / PUNTOS_DASHBOARD
    puntos.to_parquet(dest_puntos, index=False)
    logger.info("Escrito %s (%d filas).", PUNTOS_DASHBOARD, len(puntos))

    instalaciones = construir_instalaciones(cfg)
    dest_inst = REPO_ROOT / INSTALACIONES_DASHBOARD
    instalaciones.to_parquet(dest_inst, index=False)
    logger.info("Escrito %s (%d filas).", INSTALACIONES_DASHBOARD, len(instalaciones))

    geojson = construir_geojson_distritos(cfg)

    resumen = {
        "generado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "departamentos": [d["nombre"] for d in cfg["departamentos"].values()],
        "puntos_demanda_universo": int(len(puntos)),
        "puntos_demanda_muestra_enrutada": int(puntos["en_muestra_enrutada"].sum()),
        "puntos_con_poblacion_censada": int(puntos["poblacion_censada"].notna().sum()),
        "puntos_sin_acceso_enrutable": int(puntos["flag_sin_acceso_enrutable"].sum()),
        "instalaciones_total": int(len(instalaciones)),
        "instalaciones_resolutivas": int(instalaciones["es_resolutivo"].sum()),
        "instalaciones_sin_categoria": int((instalaciones["categoria_norm"] == "SIN_CATEGORIA").sum()),
        "distritos_en_geojson": (
            len(json.loads((REPO_ROOT / DISTRITOS_GEOJSON).read_text(encoding="utf-8"))["features"])
            if geojson else 0
        ),
        "artefactos": {
            "puntos": PUNTOS_DASHBOARD,
            "instalaciones": INSTALACIONES_DASHBOARD,
            "distritos": DISTRITOS_GEOJSON if geojson else None,
        },
    }
    dest_resumen = REPO_ROOT / RESUMEN_FASE4
    dest_resumen.parent.mkdir(parents=True, exist_ok=True)
    dest_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- Resumen Fase 4 ---")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("\nListo. Levanta el panel con:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
