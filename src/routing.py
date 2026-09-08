"""
routing.py — Fase 2: enrutamiento y cálculo de tiempo de viaje
================================================================

Diseñado para OSRM corriendo en Docker localmente, con un servidor HTTP por
perfil (car/bike/foot) tal como lo levanta docker/docker-compose.yml. Motor
configurable en config.md > enrutamiento.motor (osrm | osmnx) -- este módulo
implementa la ruta "osrm"; si se cambia a "osmnx" en config.md, get_engine()
debe apuntar a una implementación alternativa (no incluida aquí porque el
equipo eligió OSRM, pero la interfaz de compute_od_matrix() es la misma para
que el resto del pipeline no tenga que enterarse del cambio).

Es importable y testeable de forma aislada: cada función toma DataFrames /
listas de coordenadas y devuelve DataFrames, sin leer nada de disco excepto
para cachear resultados.

Piezas:
    - OSRMClient           cliente HTTP con reintentos y logging de progreso
    - haversine_km          distancia en línea recta (para fallback y QA)
    - sample_demand_points  muestreo estratificado a <= 5000 puntos
    - snap_points           snapea puntos a la red vial + reporte de snapping
    - compute_od_matrix     matriz origen x destino, cacheada en Parquet
    - nearest_facility      de la matriz completa, extrae la más cercana
    - compare_walk_vs_drive comparación pie vs coche pedida en la consigna
    - compare_all_modes     comparación coche/bici/pie
    - validate_deviation_factor  calibra empíricamente el factor de desvío
    - apply_fallback        asigna línea-recta*factor SOLO a lo no-enrutable,
                             marcándolo explícitamente (nunca en silencio)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config_loader import load_config, path_for

logger = logging.getLogger("routing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] routing: %(message)s")

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# Utilidades geométricas
# ---------------------------------------------------------------------------

def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Vectorizado. Usado SOLO para: (a) el fallback documentado de puntos
    no-enrutables, y (b) como referencia de QA contra las distancias
    enrutadas -- nunca como sustituto general de la distancia de red."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Muestreo (límite de 5,000 puntos de demanda)
# ---------------------------------------------------------------------------

def sample_demand_points(df: pd.DataFrame, *, col_distrito: str, col_poblacion: Optional[str],
                          cfg: Optional[dict] = None) -> pd.DataFrame:
    """Muestreo estratificado por distrito: dentro de cada distrito, se
    muestrea una fracción proporcional al total, y -si hay columna de
    población- se pondera por ella para no sub-representar centros poblados
    grandes. Si el df ya está bajo el límite, se devuelve tal cual (sin
    introducir ruido de muestreo donde no hace falta).

    Devuelve el df muestreado con una columna 'peso_muestral' = 1 / prob.
    de selección, para que las métricas de la Fase 3 puedan des-sesgar la
    agregación si se desea (implicación de error de muestreo a documentar
    en el informe).
    """
    cfg = cfg or load_config()
    limite = cfg["enrutamiento"]["muestreo"]["limite_puntos_demanda"]
    semilla = cfg["enrutamiento"]["muestreo"]["semilla_aleatoria"]

    n_total = len(df)
    if n_total <= limite:
        logger.info("Puntos de demanda (%d) ya está bajo el límite (%d); no se muestrea.", n_total, limite)
        out = df.copy()
        out["peso_muestral"] = 1.0
        return out

    frac = limite / n_total
    rng = np.random.default_rng(semilla)

    if col_poblacion and col_poblacion in df.columns:
        # Ponderado por población: probabilidad de selección proporcional a
        # la población del centro poblado dentro de su distrito.
        def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
            n_sel = max(1, round(len(g) * frac))
            n_sel = min(n_sel, len(g))
            pesos = g[col_poblacion].clip(lower=1).to_numpy(dtype=float)
            pesos = pesos / pesos.sum()
            idx = rng.choice(g.index.to_numpy(), size=n_sel, replace=False, p=pesos)
            sel = g.loc[idx].copy()
            # peso muestral = inverso de la probabilidad de inclusión aprox.
            sel["peso_muestral"] = (1.0 / (pesos[np.isin(g.index.to_numpy(), idx)] * n_sel)).round(4)
            return sel

        muestra = df.groupby(col_distrito, group_keys=False).apply(_sample_group)
    else:
        def _sample_group_uniform(g: pd.DataFrame) -> pd.DataFrame:
            n_sel = max(1, round(len(g) * frac))
            n_sel = min(n_sel, len(g))
            sel = g.sample(n=n_sel, random_state=semilla)
            sel["peso_muestral"] = len(g) / n_sel
            return sel

        muestra = df.groupby(col_distrito, group_keys=False).apply(_sample_group_uniform)

    logger.info(
        "Muestreo estratificado por distrito: %d -> %d puntos (%.1f%%). "
        "Estrategia: %s. Documentar implicación de error de muestreo en el informe (Fase 5).",
        n_total, len(muestra), 100 * len(muestra) / n_total,
        cfg["enrutamiento"]["muestreo"]["estrategia"],
    )
    return muestra.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cliente OSRM
# ---------------------------------------------------------------------------

class OSRMClient:
    """Cliente delgado sobre la API HTTP de OSRM (`/nearest`, `/table`).
    Un cliente por perfil (car/bike/foot) porque OSRM sirve cada perfil en
    su propio proceso/puerto (ver docker/docker-compose.yml).
    """

    def __init__(self, profile: str, cfg: Optional[dict] = None):
        cfg = cfg or load_config()
        self.profile = profile
        self.cfg = cfg["enrutamiento"]["osrm"]
        self.base_url = f"http://{self.cfg['host']}:{self.cfg['puertos'][profile]}"
        self.timeout = self.cfg["timeout_s"]
        self.reintentos = self.cfg["reintentos"]
        self.espera = self.cfg["espera_entre_reintentos_s"]
        self._n_requests = 0

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        last_exc = None
        for intento in range(1, self.reintentos + 1):
            try:
                self._n_requests += 1
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "Ok":
                    raise RuntimeError(f"OSRM respondió code={data.get('code')}: {data.get('message')}")
                return data
            except Exception as e:
                last_exc = e
                logger.warning("OSRM %s intento %d/%d falló: %s", self.profile, intento, self.reintentos, e)
                if intento < self.reintentos:
                    time.sleep(self.espera)
        raise RuntimeError(f"OSRM ({self.profile}) no respondió tras {self.reintentos} intentos: {last_exc}")

    def nearest(self, lon: float, lat: float) -> Optional[dict]:
        """Snapea (lon,lat) a la red vial del perfil. Devuelve None si OSRM
        no encuentra ninguna carretera cerca (radio_max implícito de OSRM),
        en vez de lanzar excepción -- el snapping fallido es un dato, no un
        error de programa."""
        try:
            data = self._get(f"/nearest/v1/{self.profile}/{lon},{lat}?number=1")
        except RuntimeError as e:
            logger.debug("Sin snap para (%s, %s) perfil=%s: %s", lon, lat, self.profile, e)
            return None
        wp = data["waypoints"][0]
        snapped_lon, snapped_lat = wp["location"]
        return {
            "snapped_lon": snapped_lon,
            "snapped_lat": snapped_lat,
            "snap_distance_m": wp["distance"],
        }

    def table(self, coords: list[tuple[float, float]], sources: list[int], destinations: list[int]) -> dict:
        """Llama a /table/v1/{profile}. coords es la lista completa
        [(lon,lat), ...]; sources/destinations son índices dentro de coords.
        Devuelve dict con 'durations' (s) y 'distances' (m), ambas matrices
        len(sources) x len(destinations), con None donde no hay ruta."""
        coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
        src_str = ";".join(map(str, sources))
        dst_str = ";".join(map(str, destinations))
        path = f"/table/v1/{self.profile}/{coord_str}?sources={src_str}&destinations={dst_str}&annotations=duration,distance"
        return self._get(path)


# ---------------------------------------------------------------------------
# Snapping con reporte
# ---------------------------------------------------------------------------

def snap_points(df: pd.DataFrame, *, col_lon: str, col_lat: str, col_id: str,
                 profile: str, cfg: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Snapea cada punto a la red del perfil dado. Devuelve (df_con_snap,
    resumen). El resumen es exactamente lo que pide la consigna: cuántos
    puntos no lograron snap y cuánto avanzó el snap medio."""
    cfg = cfg or load_config()
    client = OSRMClient(profile, cfg)
    radio_max = cfg["enrutamiento"]["snapping"]["radio_max_m"]

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        res = client.nearest(row[col_lon], row[col_lat])
        if res is None or res["snap_distance_m"] > radio_max:
            rows.append({col_id: row[col_id], "snap_ok": False, "snap_distance_m": np.nan,
                         "snapped_lon": np.nan, "snapped_lat": np.nan})
        else:
            rows.append({col_id: row[col_id], "snap_ok": True, **res})
        if i % 200 == 0:
            elapsed = time.time() - t0
            logger.info("Snapping [%s] %d/%d puntos (%.1fs transcurridos, %d requests)",
                        profile, i, len(df), elapsed, client._n_requests)

    snap_df = pd.DataFrame(rows)
    resumen = {
        "perfil": profile,
        "n_puntos": len(df),
        "n_fallidos": int((~snap_df["snap_ok"]).sum()),
        "pct_fallidos": round(100 * (~snap_df["snap_ok"]).mean(), 2),
        "distancia_media_snap_m": round(snap_df.loc[snap_df["snap_ok"], "snap_distance_m"].mean(), 1)
            if snap_df["snap_ok"].any() else None,
        "distancia_max_snap_m": round(snap_df.loc[snap_df["snap_ok"], "snap_distance_m"].max(), 1)
            if snap_df["snap_ok"].any() else None,
    }
    logger.info("Resumen snapping %s: %s", profile, resumen)
    return df.merge(snap_df, on=col_id, how="left"), resumen


# ---------------------------------------------------------------------------
# Matriz origen x destino, cacheada
# ---------------------------------------------------------------------------

def compute_od_matrix(
    demand_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    *,
    profile: str,
    col_id_demand: str,
    col_id_facility: str,
    col_lon: str = "lon",
    col_lat: str = "lat",
    cache_path: Optional[Path] = None,
    batch_size: int = 80,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Matriz COMPLETA demanda x instalación (no solo la más cercana -- la
    consigna la exige completa para poder alimentar el simulador de
    escenarios en la Fase 4). Formato largo:
        [id_demanda, id_instalacion, perfil, duracion_min, distancia_km, enrutable]

    Cacheada en Parquet: si cache_path ya existe, se lee y se devuelve sin
    volver a golpear OSRM (esto es lo que permite que el dashboard funcione
    sin motor de enrutamiento corriendo).

    OSRM /table tiene límites prácticos de tamaño de matriz por request;
    aquí se bachea por bloques de `batch_size` orígenes para no saturar una
    sola llamada, con logging de progreso y tiempo transcurrido.
    """
    cfg = cfg or load_config()
    cache_path = cache_path or path_for(f"matriz_{profile}", cfg)

    if cache_path.exists():
        logger.info("Matriz %s ya cacheada en %s, se reutiliza (borra el archivo para recalcular).",
                    profile, cache_path)
        return pd.read_parquet(cache_path)

    client = OSRMClient(profile, cfg)

    # coords combinados: primero demanda, luego instalaciones, para poder
    # referenciarlos por índice en /table
    demand_coords = list(zip(demand_df[col_lon], demand_df[col_lat]))
    facility_coords = list(zip(facility_df[col_lon], facility_df[col_lat]))
    all_coords = demand_coords + facility_coords
    dest_idx = list(range(len(demand_coords), len(all_coords)))
    facility_ids = facility_df[col_id_facility].tolist()

    registros = []
    n_batches = (len(demand_coords) + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, len(demand_coords))
        src_idx = list(range(start, end))
        batch_ids = demand_df[col_id_demand].iloc[start:end].tolist()

        try:
            data = client.table(all_coords, sources=src_idx, destinations=dest_idx)
            durations = data["durations"]  # segundos
            distances = data["distances"]  # metros
        except RuntimeError as e:
            logger.error("Batch %d/%d (perfil=%s) falló tras reintentos: %s. "
                         "Se marca como no-enrutable y se continúa (no se pierde lo ya calculado).",
                         b + 1, n_batches, profile, e)
            durations = [[None] * len(dest_idx) for _ in src_idx]
            distances = [[None] * len(dest_idx) for _ in src_idx]

        for i, demand_id in enumerate(batch_ids):
            for j, facility_id in enumerate(facility_ids):
                dur_s = durations[i][j]
                dist_m = distances[i][j]
                registros.append({
                    "id_demanda": demand_id,
                    "id_instalacion": facility_id,
                    "perfil": profile,
                    "duracion_min": (dur_s / 60.0) if dur_s is not None else np.nan,
                    "distancia_km": (dist_m / 1000.0) if dist_m is not None else np.nan,
                    "enrutable": dur_s is not None,
                })

        elapsed = time.time() - t0
        logger.info("Matriz %s: batch %d/%d listo (%d puntos) — %.1fs transcurridos, %d requests OSRM totales.",
                    profile, b + 1, n_batches, end - start, elapsed, client._n_requests)

    matriz = pd.DataFrame(registros)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matriz.to_parquet(cache_path, index=False)
    logger.info("Matriz %s escrita en caché: %s (%d filas, %.1f%% enrutable)",
                profile, cache_path, len(matriz), 100 * matriz["enrutable"].mean())
    return matriz


# ---------------------------------------------------------------------------
# Instalación más cercana, comparaciones entre modos
# ---------------------------------------------------------------------------

def nearest_facility(matrix_df: pd.DataFrame, facility_resolutive_ids: Optional[set] = None) -> pd.DataFrame:
    """De una matriz larga (id_demanda, id_instalacion, duracion_min, ...),
    extrae por punto de demanda la instalación con menor duracion_min.
    Si facility_resolutive_ids se pasa, restringe la búsqueda a esas
    instalaciones (uso: 'más cercana entre las resolutivas')."""
    df = matrix_df[matrix_df["enrutable"]].copy()
    if facility_resolutive_ids is not None:
        df = df[df["id_instalacion"].isin(facility_resolutive_ids)]
    if df.empty:
        return pd.DataFrame(columns=["id_demanda", "id_instalacion", "duracion_min", "distancia_km"])
    idx = df.groupby("id_demanda")["duracion_min"].idxmin()
    return df.loc[idx, ["id_demanda", "id_instalacion", "duracion_min", "distancia_km"]].reset_index(drop=True)


def compare_walk_vs_drive(matrix_car: pd.DataFrame, matrix_foot: pd.DataFrame,
                           facility_resolutive_ids: set) -> pd.DataFrame:
    """Requisito de la consigna: para cada punto de demanda, ¿cuál es la
    instalación resolutiva más cercana en coche vs. a pie, y difieren?"""
    car_nearest = nearest_facility(matrix_car, facility_resolutive_ids).rename(
        columns={"id_instalacion": "instalacion_car", "duracion_min": "min_car", "distancia_km": "km_car"})
    foot_nearest = nearest_facility(matrix_foot, facility_resolutive_ids).rename(
        columns={"id_instalacion": "instalacion_foot", "duracion_min": "min_foot", "distancia_km": "km_foot"})

    comp = car_nearest.merge(foot_nearest, on="id_demanda", how="outer")
    comp["misma_instalacion"] = comp["instalacion_car"] == comp["instalacion_foot"]
    comp["ratio_foot_car"] = comp["min_foot"] / comp["min_car"].replace(0, np.nan)

    n = len(comp)
    n_dif = int((~comp["misma_instalacion"].fillna(False)).sum())
    n_inaccesible_foot = int(comp["min_foot"].isna().sum())
    logger.info(
        "Comparación pie vs coche: %d/%d puntos (%.1f%%) tienen distinta instalación más cercana "
        "según el modo. %d puntos (%.1f%%) son inalcanzables a pie dentro de la red mapeada.",
        n_dif, n, 100 * n_dif / n if n else 0, n_inaccesible_foot, 100 * n_inaccesible_foot / n if n else 0,
    )
    return comp


def compare_all_modes(matrices: dict[str, pd.DataFrame], facility_resolutive_ids: set) -> pd.DataFrame:
    """matrices = {'car': df, 'bike': df, 'foot': df}. Devuelve un df ancho
    con el tiempo a la instalación resolutiva más cercana bajo cada perfil,
    más la razón pie/coche y bici/coche por punto de demanda."""
    wide = None
    for perfil, m in matrices.items():
        nearest = nearest_facility(m, facility_resolutive_ids)[["id_demanda", "duracion_min"]]
        nearest = nearest.rename(columns={"duracion_min": f"min_{perfil}"})
        wide = nearest if wide is None else wide.merge(nearest, on="id_demanda", how="outer")

    if "min_foot" in wide.columns and "min_car" in wide.columns:
        wide["ratio_foot_car"] = wide["min_foot"] / wide["min_car"].replace(0, np.nan)
    if "min_bike" in wide.columns and "min_car" in wide.columns:
        wide["ratio_bike_car"] = wide["min_bike"] / wide["min_car"].replace(0, np.nan)
    return wide


# ---------------------------------------------------------------------------
# Fallback para puntos no enrutables + calibración empírica del factor
# ---------------------------------------------------------------------------

def validate_deviation_factor(matrix_df: pd.DataFrame, demand_df: pd.DataFrame, facility_df: pd.DataFrame,
                               col_id_demand: str, col_id_facility: str,
                               col_lon: str = "lon", col_lat: str = "lat") -> dict:
    """Para los pares (demanda, instalación) que SÍ tienen ruta, compara
    distancia_km enrutada contra la distancia Haversine equivalente. El
    cociente mediano (distancia_red / distancia_recta) es el factor de
    desvío empírico -- esto reemplaza el valor por defecto puesto a mano en
    config.md (enrutamiento.fallback_no_enrutable.factor_desvio_default),
    que debe recalibrarse con esta función antes de usarse en el informe.
    """
    routed = matrix_df[matrix_df["enrutable"]].copy()
    d = demand_df.set_index(col_id_demand)[[col_lon, col_lat]]
    f = facility_df.set_index(col_id_facility)[[col_lon, col_lat]]

    routed = routed.join(d.rename(columns={col_lon: "lon_d", col_lat: "lat_d"}), on="id_demanda")
    routed = routed.join(f.rename(columns={col_lon: "lon_f", col_lat: "lat_f"}), on="id_instalacion")
    routed = routed.dropna(subset=["lon_d", "lat_d", "lon_f", "lat_f", "distancia_km"])

    routed["dist_haversine_km"] = haversine_km(
        routed["lon_d"].to_numpy(), routed["lat_d"].to_numpy(),
        routed["lon_f"].to_numpy(), routed["lat_f"].to_numpy(),
    )
    routed = routed[routed["dist_haversine_km"] > 0.01]  # evita división por ~0
    routed["ratio"] = routed["distancia_km"] / routed["dist_haversine_km"]

    resumen = {
        "n_pares": len(routed),
        "factor_mediano": round(routed["ratio"].median(), 3),
        "factor_p25": round(routed["ratio"].quantile(0.25), 3),
        "factor_p75": round(routed["ratio"].quantile(0.75), 3),
        "factor_medio": round(routed["ratio"].mean(), 3),
    }
    logger.info("Factor de desvío empírico (red/línea-recta): %s", resumen)
    return resumen


def apply_fallback(nearest_df: pd.DataFrame, demand_df: pd.DataFrame, facility_df: pd.DataFrame,
                    unroutable_demand_ids: list, col_id_demand: str, col_id_facility: str,
                    factor_desvio: float, velocidad_kmh: float = 40.0,
                    col_lon: str = "lon", col_lat: str = "lat") -> pd.DataFrame:
    """Asigna a los puntos NO enrutables una estimación lineal *solo si*
    config.md > fallback_no_enrutable.permitir_linea_recta es True, y
    siempre con una columna `estimado_por_fallback=True` explícita -- nunca
    se mezcla en silencio con los puntos realmente enrutados.

    Se asigna la instalación resolutiva geográficamente más cercana en
    línea recta, con distancia = haversine * factor_desvio (calibrado con
    validate_deviation_factor) y tiempo = distancia / velocidad_kmh.
    """
    cfg = load_config()
    if not cfg["enrutamiento"]["fallback_no_enrutable"]["permitir_linea_recta"]:
        logger.warning("%d puntos no enrutables NO reciben fallback (deshabilitado en config.md); "
                       "quedan como NaN.", len(unroutable_demand_ids))
        return nearest_df

    d = demand_df.set_index(col_id_demand)
    f = facility_df.set_index(col_id_facility)
    filas = []
    for did in unroutable_demand_ids:
        if did not in d.index:
            continue
        lon_d, lat_d = d.loc[did, col_lon], d.loc[did, col_lat]
        dists = haversine_km(
            np.full(len(f), lon_d), np.full(len(f), lat_d),
            f[col_lon].to_numpy(), f[col_lat].to_numpy(),
        )
        j = int(np.argmin(dists))
        dist_recta = dists[j]
        dist_estim = dist_recta * factor_desvio
        filas.append({
            "id_demanda": did,
            "id_instalacion": f.index[j],
            "duracion_min": round(60 * dist_estim / velocidad_kmh, 1),
            "distancia_km": round(dist_estim, 2),
            "estimado_por_fallback": True,
        })

    fallback_df = pd.DataFrame(filas)
    nearest_df = nearest_df.copy()
    nearest_df["estimado_por_fallback"] = False
    out = pd.concat([nearest_df, fallback_df], ignore_index=True)
    logger.warning(
        "%d/%d puntos no enrutables recibieron estimación lineal (factor=%.3f, "
        "velocidad asumida=%.0f km/h). Marcados con estimado_por_fallback=True: "
        "NO deben tratarse igual que un tiempo enrutado real en el análisis de Fase 3.",
        len(fallback_df), len(unroutable_demand_ids), factor_desvio, velocidad_kmh,
    )
    return out


if __name__ == "__main__":
    # Prueba de humo sin red: valida que las funciones puramente numéricas
    # (haversine, muestreo, nearest_facility) se comporten bien con datos
    # sintéticos. La parte que llama a OSRM real se prueba en tests/test_routing.py
    # con un servidor OSRM local levantado (ver README).
    demo_matrix = pd.DataFrame({
        "id_demanda": ["D1", "D1", "D2", "D2"],
        "id_instalacion": ["F1", "F2", "F1", "F2"],
        "perfil": ["car"] * 4,
        "duracion_min": [12.0, 30.0, 45.0, 8.0],
        "distancia_km": [5.0, 15.0, 20.0, 4.0],
        "enrutable": [True, True, True, True],
    })
    print(nearest_facility(demo_matrix, facility_resolutive_ids={"F1", "F2"}))
    print("haversine Lima-Piura (km aprox):",
          round(float(haversine_km(np.array([-77.03]), np.array([-12.05]), np.array([-80.63]), np.array([-5.19]))[0]), 1))
