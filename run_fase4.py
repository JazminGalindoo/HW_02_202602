"""
run_fase4.py — Fase 4 (preparación): artefactos que consume el panel
=====================================================================

El panel de Streamlit (`app.py`) tiene que abrir en segundos, sin OSRM
levantado, sin Docker y sin geopandas instalado. Para eso este script deja
pre-cocinados cinco artefactos y un resumen de trazabilidad:

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

    data/processed/mejoras_candidatos.parquet
        Insumo del SIMULADOR DE ESCENARIOS: para cada par (centro poblado,
        establecimiento I-3/I-4 candidato a ascenso) en el que ascender ese
        establecimiento MEJORARÍA el tiempo actual, el tiempo estimado hacia
        él. Solo se guardan los pares que mejoran (245 mil de 953 mil
        posibles): los demás no cambian ningún indicador y ocuparían disco
        para nada. Ver `construir_mejoras_candidatos` para el origen de los
        tiempos y sus supuestos.

    data/outputs/calibracion_linea_recta.csv  (+ calibracion_pares_muestra.csv)
        Comparación distancia de red vs. línea recta por departamento
        (Fase 2, `routing.validate_deviation_factor` extendido), y una muestra
        de 20,000 pares para graficarla. Alimenta el simulador de escenarios
        y la sección de discusión del informe.

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
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import REPO_ROOT, load_config, path_for
from src.dashboard_data import (
    CALIBRACION_PARES, CALIBRACION_RECTA, DISTRITOS_GEOJSON, INSTALACIONES_DASHBOARD, MEJORAS_CANDIDATOS,
    PUNTOS_DASHBOARD, RESUMEN_FASE4, assign_bands,
)
from src.metrics import access_time, attach_sample_weights, classify_urban_rural
from src.routing import haversine_km, sample_demand_points, validate_deviation_factor

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


def calibrar_linea_recta(cfg: dict) -> pd.DataFrame:
    """Compara distancia de red contra distancia en línea recta, por
    departamento, sobre los 284,432 pares que Fase 2 sí logró enrutar.

    Extiende `routing.validate_deviation_factor` (que da el factor global)
    con dos cosas que el simulador y el informe necesitan:

      - el factor de desvío POR DEPARTAMENTO (la red de Loreto desvía más
        que la de Ayacucho, y usar un único factor nacional mezcla peras con
        manzanas), y
      - los `minutos por kilómetro en línea recta`, que es el factor de
        desvío y la velocidad de la red combinados en un solo número. Es el
        parámetro que usa el simulador para estimar tiempos hacia
        establecimientos que Fase 2 nunca enrutó.

    Se usa la MEDIANA y no la media porque la distribución del cociente
    tiene cola larga (hay pares con desvíos de 40x cuando el destino está al
    otro lado de un río)."""
    demanda = pd.read_parquet(path_for("demanda_processed", cfg))
    ipress = pd.read_parquet(path_for("ipress_processed", cfg)).rename(
        columns={"ESTE": "lon", "NORTE": "lat", "COD_IPRESS": "id"})
    resolutivas = ipress[ipress["es_resolutivo"]]
    matriz = pd.read_parquet(path_for("matriz_car", cfg))
    muestra = sample_demand_points(demanda, col_distrito="distrito", col_poblacion=None, cfg=cfg)

    global_ = validate_deviation_factor(matriz, muestra, resolutivas, "id", "id")

    pares = matriz[matriz["enrutable"]].merge(
        muestra[["id", "lon", "lat", "DEP"]], left_on="id_demanda", right_on="id",
    ).merge(
        resolutivas[["id", "lon", "lat"]], left_on="id_instalacion", right_on="id",
        suffixes=("_d", "_f"),
    )
    pares["dist_recta_km"] = haversine_km(
        pares["lon_d"].to_numpy(), pares["lat_d"].to_numpy(),
        pares["lon_f"].to_numpy(), pares["lat_f"].to_numpy(),
    )
    # Pares con origen y destino casi encima: el cociente se dispara por
    # división entre casi cero y no informa sobre la forma de la red.
    pares = pares[(pares["dist_recta_km"] > 0.5) & (pares["duracion_min"] > 0)]
    pares = pares.assign(
        factor_desvio=pares["distancia_km"] / pares["dist_recta_km"],
        velocidad_kmh=pares["distancia_km"] / (pares["duracion_min"] / 60),
        min_por_km_recta=pares["duracion_min"] / pares["dist_recta_km"],
    )

    filas = []
    for dep, g in pares.groupby("DEP"):
        filas.append({
            "departamento": dep,
            "n_pares": len(g),
            "factor_desvio_p25": round(g["factor_desvio"].quantile(0.25), 3),
            "factor_desvio_mediano": round(g["factor_desvio"].median(), 3),
            "factor_desvio_p75": round(g["factor_desvio"].quantile(0.75), 3),
            "velocidad_mediana_kmh": round(g["velocidad_kmh"].median(), 1),
            "min_por_km_recta_mediano": round(g["min_por_km_recta"].median(), 3),
        })
    filas.append({
        "departamento": "TODOS",
        "n_pares": int(global_["n_pares"]),
        "factor_desvio_p25": float(global_["factor_p25"]),
        "factor_desvio_mediano": float(global_["factor_mediano"]),
        "factor_desvio_p75": float(global_["factor_p75"]),
        "velocidad_mediana_kmh": round(pares["velocidad_kmh"].median(), 1),
        "min_por_km_recta_mediano": round(pares["min_por_km_recta"].median(), 3),
    })
    calibracion = pd.DataFrame(filas)

    dest = REPO_ROOT / CALIBRACION_RECTA
    dest.parent.mkdir(parents=True, exist_ok=True)
    calibracion.to_csv(dest, index=False)

    # Muestra de pares para la figura del informe: 284,432 puntos en un
    # gráfico de dispersión son un borrón negro y 12 MB de PDF vectorial.
    # Semilla fija para que la figura no cambie entre corridas.
    muestra_pares = pares.sample(
        n=min(20_000, len(pares)), random_state=cfg["enrutamiento"]["muestreo"]["semilla_aleatoria"],
    )[["DEP", "dist_recta_km", "distancia_km", "duracion_min", "factor_desvio", "velocidad_kmh"]]
    muestra_pares.round(4).to_csv(REPO_ROOT / CALIBRACION_PARES, index=False)
    logger.info(
        "Calibración línea recta vs. red: factor de desvío mediano global %.2f "
        "(config.md trae %.2f por defecto). Por departamento: %s",
        global_["factor_mediano"],
        cfg["enrutamiento"]["fallback_no_enrutable"]["factor_desvio_default"],
        {f["departamento"]: f["factor_desvio_mediano"] for f in filas[:-1]},
    )
    return calibracion


def construir_mejoras_candidatos(cfg: dict, puntos: pd.DataFrame, instalaciones: pd.DataFrame,
                                  calibracion: pd.DataFrame) -> pd.DataFrame:
    """Insumo del simulador de escenarios: tiempo estimado de cada centro
    poblado muestreado hacia cada establecimiento I-3/I-4 de su departamento.

    ORIGEN DE LOS TIEMPOS Y POR QUÉ SON ESTIMADOS
    ---------------------------------------------
    La matriz OD de Fase 2 se calculó contra las instalaciones RESOLUTIVAS
    (58), no contra las 607 candidatas a ascenso, así que no contiene el
    tiempo hacia un I-3 cualquiera. Hay dos formas de conseguirlo:

      (a) Recalcular con OSRM la matriz demanda x candidatos. Si ese archivo
          existe (`data/processed/matriz_car_candidatos.parquet`, ver el
          README para generarlo con Docker levantado), esta función lo usa y
          marca los tiempos como `osrm`.

      (b) Si no existe --- el caso por defecto, para que el panel funcione en
          una máquina sin Docker ---, se estima con la calibración empírica
          de `calibrar_linea_recta`: distancia en línea recta multiplicada
          por los minutos-por-kilómetro-recto medianos DEL DEPARTAMENTO, que
          ya incorporan el desvío de la red y su velocidad típica. Los
          tiempos se marcan como `estimado` y el panel lo dice en pantalla.

    El supuesto que hay que tener presente al leer el simulador: la
    calibración se hizo sobre viajes largos hacia hospitales, y un viaje
    corto hacia el I-3 del pueblo vecino puede ser proporcionalmente más
    lento (vías locales) o más rápido (no hay que rodear un accidente
    geográfico). En Loreto, además, la calibración hereda el problema de
    fondo: describe una red vial que en la práctica no se usa.

    Solo se conservan los pares que MEJORAN el tiempo actual del punto. Los
    demás son irrelevantes por construcción del indicador --- t_min es un
    mínimo, así que un candidato más lento que el hospital actual no cambia
    nada --- y guardarlos multiplicaría por cuatro el tamaño del archivo."""
    candidatos_cat = ["I-3", "I-4"]
    candidatos = instalaciones[instalaciones["categoria_norm"].isin(candidatos_cat)].copy()
    muestra = puntos[puntos["en_muestra_enrutada"]]

    ruta_osrm = REPO_ROOT / "data/processed/matriz_car_candidatos.parquet"
    if ruta_osrm.exists():
        matriz = pd.read_parquet(ruta_osrm)
        mejoras = matriz.rename(columns={
            "id_instalacion": "id_candidato", "duracion_min": "t_candidato_min",
        })[["id_demanda", "id_candidato", "t_candidato_min"]]
        mejoras["fuente"] = "osrm"
        actual = muestra.set_index("id")["t_min"]
        mejoras = mejoras[
            mejoras["t_candidato_min"]
            < mejoras["id_demanda"].map(actual).fillna(np.inf)
        ]
        logger.info("Simulador: usando la matriz OSRM demanda x candidatos (%d pares que mejoran).",
                    len(mejoras))
    else:
        min_por_km = dict(zip(calibracion["departamento"],
                              calibracion["min_por_km_recta_mediano"]))
        bloques = []
        for dep, grupo in muestra.groupby("DEP"):
            cand_dep = candidatos[candidatos["DEPARTAMENTO"] == dep]
            if cand_dep.empty:
                continue
            factor = float(min_por_km.get(dep, min_por_km["TODOS"]))
            # Matriz densa por departamento (el mayor es 5,002 x 292): cabe
            # de sobra en memoria y evita un bucle por par en Python.
            distancias = haversine_km(
                grupo["lon"].to_numpy()[:, None], grupo["lat"].to_numpy()[:, None],
                cand_dep["lon"].to_numpy()[None, :], cand_dep["lat"].to_numpy()[None, :],
            )
            # Se redondea ANTES de comparar, no al guardar: si no, un par
            # cuyo tiempo real mejora por 3 milésimas de minuto se guarda
            # redondeado por encima del tiempo actual y el archivo deja de
            # cumplir su propia invariante ("solo pares que mejoran").
            tiempos = np.round(distancias * factor, 2)
            actual = grupo["t_min"].to_numpy()[:, None]
            mejora = tiempos < np.where(np.isnan(actual), np.inf, actual)
            filas_i, filas_j = np.nonzero(mejora)
            bloques.append(pd.DataFrame({
                "id_demanda": grupo["id"].to_numpy()[filas_i],
                "id_candidato": cand_dep["COD_IPRESS"].to_numpy()[filas_j],
                "t_candidato_min": tiempos[filas_i, filas_j],
                "fuente": "estimado",
            }))
        mejoras = pd.concat(bloques, ignore_index=True) if bloques else pd.DataFrame(
            columns=["id_demanda", "id_candidato", "t_candidato_min", "fuente"])
        logger.info(
            "Simulador: sin matriz OSRM de candidatos; tiempos estimados con la calibración "
            "por departamento (%s min por km en línea recta). %d pares que mejoran, "
            "%d candidatos con algún impacto de %d.",
            {d: round(v, 2) for d, v in min_por_km.items() if d != "TODOS"},
            len(mejoras), mejoras["id_candidato"].nunique(), len(candidatos),
        )

    dest = REPO_ROOT / MEJORAS_CANDIDATOS
    dest.parent.mkdir(parents=True, exist_ok=True)
    mejoras.to_parquet(dest, index=False)
    return mejoras


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

    calibracion = calibrar_linea_recta(cfg)
    mejoras = construir_mejoras_candidatos(cfg, puntos, instalaciones, calibracion)
    logger.info("Escrito %s (%d pares).", MEJORAS_CANDIDATOS, len(mejoras))

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
        "simulador_candidatos": int(mejoras["id_candidato"].nunique()),
        "simulador_pares_que_mejoran": int(len(mejoras)),
        "simulador_fuente_tiempos": (
            sorted(mejoras["fuente"].unique().tolist()) if len(mejoras) else []
        ),
        "factor_desvio_mediano_empirico": float(
            calibracion.loc[calibracion["departamento"] == "TODOS", "factor_desvio_mediano"].iloc[0]
        ),
        "artefactos": {
            "puntos": PUNTOS_DASHBOARD,
            "instalaciones": INSTALACIONES_DASHBOARD,
            "distritos": DISTRITOS_GEOJSON if geojson else None,
            "mejoras_candidatos": MEJORAS_CANDIDATOS,
            "calibracion_linea_recta": CALIBRACION_RECTA,
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
