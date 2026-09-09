"""
metrics.py — Fase 3: métricas y análisis de accesibilidad
=============================================================

Cada métrica es una función pura: toma DataFrame(s) y devuelve DataFrame(s).
Ninguna lógica de negocio de métricas vive en el dashboard (Fase 4) -- el
dashboard solo lee los CSV/Parquet que este módulo produce.

Piezas:
    - classify_urban_rural   regla urbano/rural explícita sobre CAPITAL (Tarea 2)
    - attach_sample_weights  reproduce la muestra+peso_muestral de Fase 2
                              (routing.sample_demand_points, misma semilla),
                              necesaria para no sesgar los agregados
                              poblacionales hacia la muestra enrutada
    - access_time            t_min(i) por punto de demanda muestreado
    - coverage_bands         % población en bandas 0-30/30-60/60-120/>120 min
    - weighted_mean_access   t_min medio ponderado por población, por nivel
    - critical_gap_ranking   distritos/provincias peor rankeados
    - gini_access            Gini ponderado por población + curva de Lorenz
    - urban_rural_contrast   contraste de acceso urbano vs. rural
    - access_vs_altitude     cruce t_min vs. altitud (Z), con nota de causalidad
    - weighted_median_access mediana ponderada por población (Fase 4)
    - population_within      población bajo/sobre un umbral de minutos (Fase 4)
    - apply_upgrades         escenario ``¿y si se asciende este I-3 a II-1?''
                              (Fase 4; usa la matriz completa de Fase 2)
    - ranking_upgrade_gain   ganancia individual de cada candidato a ascenso

Esquema esperado de `population_df` en todas las funciones que lo reciben:
al menos las columnas de demanda_con_poblacion.parquet (salida de
population.py) -- en particular `id`, `poblacion_censada`, `distrito`
(ubigeo de 6 dígitos), `PROV`, `DEP` -- y opcionalmente `peso_muestral`
(ver attach_sample_weights) y `zona` (ver classify_urban_rural).

IMPORTANTE sobre representatividad: las matrices de enrutamiento (Fase 2)
solo cubren la muestra estratificada de <= 5,000 puntos de
`routing.sample_demand_points`, no los ~19,460 puntos de demanda_clean. Toda
métrica aquí opera sobre esa muestra; se reporta explícitamente qué
fracción de la población total del universo queda representada, en vez de
insinuar que las cifras describen a los 3 departamentos completos sin
matiz.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.config_loader import load_config
from src.routing import nearest_facility, sample_demand_points

logger = logging.getLogger("metrics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Ver la nota equivalente en population.py: no usamos logging.basicConfig()
    # porque el primer módulo importado en el proceso (acquisition/validation/
    # routing) ya lo llamó con su propio nombre hardcodeado en el formato, y
    # basicConfig() es un no-op en llamadas subsecuentes.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] metrics: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

_NIVEL_A_UBIGEO_LEN = {"departamento": 2, "provincia": 4, "distrito": 6}
_NIVEL_A_COL_NOMBRE = {"departamento": "DEP", "provincia": "PROV", "distrito": "DIST"}


# ---------------------------------------------------------------------------
# Tarea 2 — Clasificación urbano / rural
# ---------------------------------------------------------------------------

def classify_urban_rural(df: pd.DataFrame, col_capital: str = "CAPITAL") -> pd.DataFrame:
    """Añade la columna 'zona' ∈ {'urbano','rural'} a una copia de df.

    REGLA (explícita, sobre la columna CAPITAL de demanda_clean):
        CAPITAL == '0'            -> 'rural'
        CAPITAL ∈ {'1','2','3'}   -> 'urbano'

    Verificado empíricamente contra la jerarquía administrativa real de los
    3 departamentos del proyecto: CAPITAL codifica, con exclusión mutua
    jerárquica (un centro poblado que ya es capital departamental NO se
    vuelve a marcar como capital provincial/distrital), el nivel de capital
    del centro poblado:
        '0' = no es capital de nada
        '1' = capital departamental  (exactamente 3 en este proyecto: 1 por depto)
        '2' = capital provincial     (conteo por depto ≈ n_provincias - 1,
                                       porque la provincia cuya capital
                                       coincide con la capital departamental
                                       ya quedó marcada '1')
        '3' = capital distrital      (mismo efecto de exclusión un nivel abajo)

    JUSTIFICACIÓN de usar "es capital (de distrito/provincia/departamento)"
    como proxy de urbano/rural: demanda_clean (derivado del shapefile de
    centros poblados de SIGMED) NO trae la variable oficial de área urbana/
    rural del INEI (que se define por densidad poblacional en manzanas
    censales contiguas, no está en este extracto). "Ser capital político-
    administrativa" es la única variable disponible que aproxima
    concentración urbana: las capitales concentran típicamente servicios,
    infraestructura vial y la mayor densidad poblacional de su jurisdicción.

    LIMITACIONES documentadas (a citar en el informe, no a esconder):
      - Es un proxy administrativo, NO la definición oficial INEI de área
        urbana (que usa densidad/contigüidad de manzanas, no disponible aquí).
      - Una capital distrital de un distrito andino/amazónico muy pequeño
        puede tener población mínima (ver Tarea 1: hay centros poblados con
        0 habitantes censados) y no ser "urbana" en sentido socioeconómico.
      - Un centro poblado NO capital pero grande/conurbado con una capital
        cercana queda clasificado como 'rural' aunque funcionalmente sea
        periurbano/urbano.

    Consistente con el criterio ya usado en run_fase2.py (comparación pie
    vs. coche, Fase 2: "Urbano = cualquier nivel de capital") -- se llegó a
    la misma regla de forma independiente al verificar CAPITAL contra la
    jerarquía administrativa real de los 3 departamentos.
    """
    if col_capital not in df.columns:
        raise KeyError(f"classify_urban_rural: falta la columna '{col_capital}' en df")

    es_nulo = df[col_capital].isna()  # detectar ANTES de convertir a str: astype(str)
                                       # convierte None -> "None" (deja de ser nulo)
    n_nulos = int(es_nulo.sum())
    if n_nulos:
        # No debería ocurrir en los datos actuales (CAPITAL siempre viene
        # poblado con '0'/'1'/'2'/'3'), pero si algún rerun futuro trae
        # nulos, se advierte en vez de decidir en silencio si cuentan como
        # urbano o rural -- por defecto quedan como 'rural' (lado
        # conservador: no asumir capital sin evidencia).
        logger.warning(
            "classify_urban_rural: %d filas con '%s' nulo -- se clasifican como 'rural' "
            "por defecto (revisar la fuente si esto es inesperado).",
            n_nulos, col_capital,
        )
    capital_str = df[col_capital].astype(str).str.strip()
    es_capital = ~es_nulo & (capital_str != "0")
    out = df.assign(zona=np.where(es_capital, "urbano", "rural"))

    n_urbano = int((out["zona"] == "urbano").sum())
    n_total = len(out)
    logger.info(
        "classify_urban_rural: %d/%d (%.1f%%) centros poblados clasificados como 'urbano' "
        "(CAPITAL != '0'); regla documentada en el docstring de esta función -- proxy "
        "administrativo, no la definición oficial INEI de área urbana.",
        n_urbano, n_total, 100 * n_urbano / n_total if n_total else 0,
    )
    return out


# ---------------------------------------------------------------------------
# Pesos de muestreo (reproduce Fase 2 para no sesgar agregados poblacionales)
# ---------------------------------------------------------------------------

def attach_sample_weights(demanda_df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Reproduce EXACTAMENTE la muestra estratificada por distrito que
    run_fase2.py usó para calcular las matrices OD (misma semilla,
    col_poblacion=None -- Fase 2 corrió antes de que existiera población por
    centro poblado, así que el muestreo fue uniforme dentro de cada
    distrito, no ponderado por población).

    Devuelve demanda_df con una columna adicional 'peso_muestral': NaN para
    los puntos que NO quedaron en la muestra enrutada (no tienen fila en
    matriz_car/foot/bike y por tanto no pueden aparecer en access_df), y el
    peso de diseño (1/prob. de inclusión aprox.) para los que sí.

    Sin esto, agregar poblacion_censada directamente sobre la muestra
    subestima sistemáticamente a los distritos con muchos centros poblados
    pequeños (sub-muestreados proporcionalmente igual que los grandes,
    porque la selección fue uniforme, no ponderada)."""
    cfg = cfg or load_config()
    muestra = sample_demand_points(demanda_df, col_distrito="distrito", col_poblacion=None, cfg=cfg)
    pesos = muestra[["id", "peso_muestral"]]
    out = demanda_df.merge(pesos, on="id", how="left")
    logger.info(
        "attach_sample_weights: %d/%d puntos de demanda quedan en la muestra enrutada de Fase 2 "
        "(peso_muestral asignado); el resto no tiene fila en las matrices OD.",
        pesos["id"].notna().sum(), len(demanda_df),
    )
    return out


def _effective_weight(df: pd.DataFrame) -> pd.Series:
    """poblacion_censada * peso_muestral si hay peso de diseño disponible
    (des-sesga la muestra), si no, poblacion_censada a secas. NaN en
    cualquiera de los dos factores se trata como peso 0 -- excluye esa fila
    del numerador Y denominador de cualquier promedio/porcentaje ponderado,
    en vez de que un NaN se propague silenciosamente a todo el resultado."""
    poblacion = df["poblacion_censada"].fillna(0.0)
    if "peso_muestral" in df.columns:
        peso = df["peso_muestral"].fillna(0.0)
        return poblacion * peso
    return poblacion


def _join_by_id(access_df: pd.DataFrame, other_df: pd.DataFrame, other_id_col: str = "id") -> pd.DataFrame:
    """Une access_df (columna 'id_demanda') a otro DataFrame por su columna
    de id (por defecto 'id', el esquema de demanda_clean/demanda_con_poblacion).
    No descarta puntos de access_df sin match -- se conservan con las
    columnas de other_df en NaN, y se advierte si hay alguno (no debería
    pasar: todo id_demanda de la matriz OD viene de demanda_clean)."""
    cols_other = [c for c in other_df.columns if c == other_id_col or c not in access_df.columns]
    merged = access_df.merge(
        other_df[cols_other], left_on="id_demanda", right_on=other_id_col, how="left"
    )
    n_sin_match = merged[other_id_col].isna().sum()
    if n_sin_match:
        logger.warning(
            "%d puntos de access_df no matchean por id contra el DataFrame provisto "
            "(esperado: 0, todo id_demanda debería existir en demanda_clean).",
            n_sin_match,
        )
    return merged


# ---------------------------------------------------------------------------
# access_time
# ---------------------------------------------------------------------------

def access_time(matrix_df: pd.DataFrame, facility_resolutive_ids: set) -> pd.DataFrame:
    """t_min(i) por cada punto de demanda presente en matrix_df (el universo
    de esta métrica ES el universo enrutado, típicamente la muestra
    estratificada de Fase 2 -- no los ~19,460 de demanda_clean).

    Para cada id_demanda, t_min = duración a la instalación RESOLUTIVA más
    cercana enrutable. Si ninguna instalación resolutiva es alcanzable
    dentro de la red mapeada, t_min=NaN y flag_sin_acceso_enrutable=True --
    no se descarta el punto (mismo patrón flag_* que validation.py)."""
    universo = matrix_df[["id_demanda"]].drop_duplicates().reset_index(drop=True)
    nearest = nearest_facility(matrix_df, facility_resolutive_ids)[
        ["id_demanda", "id_instalacion", "duracion_min"]
    ].rename(columns={"duracion_min": "t_min"})

    merged = universo.merge(nearest, on="id_demanda", how="left")
    out = merged.assign(flag_sin_acceso_enrutable=merged["t_min"].isna())

    n_total = len(out)
    n_sin_acceso = int(out["flag_sin_acceso_enrutable"].sum())
    logger.info(
        "access_time: %d puntos de demanda en la matriz; %d (%.1f%%) sin ninguna instalación "
        "resolutiva enrutable (t_min=NaN, conservados con flag_sin_acceso_enrutable=True).",
        n_total, n_sin_acceso, 100 * n_sin_acceso / n_total if n_total else 0,
    )
    return out


# ---------------------------------------------------------------------------
# coverage_bands
# ---------------------------------------------------------------------------

def coverage_bands(access_df: pd.DataFrame, population_df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """% de población (ponderada, ver _effective_weight) en cada banda de
    t_min: 0-30 / 30-60 / 60-120 / >120 minutos (bordes en
    config.md > metricas.bandas_acceso_min), más una banda
    'sin_acceso_enrutable' para los puntos con t_min=NaN (población
    conocida pero sin instalación resolutiva alcanzable -- se cuentan
    aparte, NO se excluyen del reporte).

    El % es sobre la población total CONOCIDA de la muestra enrutada (los
    puntos con población NaN -- ver Tarea 1 -- pesan 0 y por tanto no
    entran ni al numerador ni al denominador; se reporta ese excluido
    aparte por log, no se pierde en silencio)."""
    cfg = cfg or load_config()
    bordes = cfg["metricas"]["bandas_acceso_min"]  # p.ej. [30, 60, 120]

    merged = _join_by_id(access_df, population_df)
    merged = merged.assign(peso=_effective_weight(merged))

    n_sin_poblacion_conocida = int(merged["poblacion_censada"].isna().sum())

    bins = [-1e-9] + list(bordes) + [np.inf]
    labels = [f"0-{bordes[0]}"] + [f"{bordes[i]}-{bordes[i+1]}" for i in range(len(bordes) - 1)] + [f">{bordes[-1]}"]
    merged = merged.assign(banda=pd.cut(merged["t_min"], bins=bins, labels=labels))

    sin_acceso = merged.loc[merged["flag_sin_acceso_enrutable"], "peso"].sum()
    por_banda = merged.dropna(subset=["banda"]).groupby("banda", observed=True)["peso"].sum()

    filas = [{"banda": b, "poblacion": float(por_banda.get(b, 0.0))} for b in labels]
    filas.append({"banda": "sin_acceso_enrutable", "poblacion": float(sin_acceso)})
    reporte = pd.DataFrame(filas)

    poblacion_total_conocida = reporte["poblacion"].sum()
    if poblacion_total_conocida > 0:
        pct = (100 * reporte["poblacion"] / poblacion_total_conocida).round(2)
    else:
        pct = pd.Series(0.0, index=reporte.index)
    reporte = reporte.assign(pct_poblacion=pct)

    logger.info(
        "coverage_bands: población total considerada (ponderada, muestra enrutada) = %.0f. "
        "%d puntos de la muestra tienen población desconocida (excluidos, peso 0). "
        "Bandas: %s",
        poblacion_total_conocida, n_sin_poblacion_conocida,
        {r["banda"]: round(r["pct_poblacion"], 1) for r in reporte.to_dict("records")},
    )
    return reporte


# ---------------------------------------------------------------------------
# weighted_mean_access
# ---------------------------------------------------------------------------

def _nivel_columnas(merged: pd.DataFrame, level: str) -> pd.DataFrame:
    if level not in _NIVEL_A_UBIGEO_LEN:
        raise ValueError(f"level debe ser uno de {list(_NIVEL_A_UBIGEO_LEN)}, recibido: {level!r}")
    largo = _NIVEL_A_UBIGEO_LEN[level]
    col_nombre = _NIVEL_A_COL_NOMBRE[level]
    nivel_id = merged["distrito"].astype(str).str.zfill(6).str[:largo]
    nivel_nombre = merged[col_nombre] if col_nombre in merged.columns else nivel_id
    out = merged.assign(nivel_id=nivel_id, nivel_nombre=nivel_nombre)
    return out


def weighted_mean_access(
    access_df: pd.DataFrame, population_df: pd.DataFrame, level: str = "distrito", cfg: Optional[dict] = None
) -> pd.DataFrame:
    """t_min medio ponderado por población, agregado a nivel 'distrito',
    'provincia' o 'departamento' (agrupado por prefijo de ubigeo: 2/4/6
    dígitos -- más robusto que agrupar por nombre de texto, que puede
    repetirse entre departamentos distintos).

    Devuelve una fila por unidad geográfica con:
        nivel_id, nivel_nombre, t_min_medio_ponderado, poblacion_con_acceso,
        poblacion_sin_acceso_enrutable, pct_poblacion_con_acceso, n_puntos

    t_min_medio_ponderado se calcula SOLO sobre puntos con acceso enrutable
    conocido (t_min no nulo); poblacion_sin_acceso_enrutable expone
    explícitamente cuánta población de esa unidad quedó fuera de ese
    promedio, para que no se lea como "el 100% de la población tiene este
    tiempo medio" cuando en realidad una parte no tiene ruta calculada."""
    cfg = cfg or load_config()
    merged = _join_by_id(access_df, population_df)
    merged = _nivel_columnas(merged, level)
    merged = merged.assign(peso=_effective_weight(merged))

    con_acceso = merged.dropna(subset=["t_min"])
    filas = []
    for nivel_id, g in merged.groupby("nivel_id"):
        g_con_acceso = g.dropna(subset=["t_min"])
        peso_con_acceso = g_con_acceso["peso"].sum()
        peso_sin_acceso = g.loc[g["flag_sin_acceso_enrutable"].fillna(False), "peso"].sum()
        t_medio = (
            np.average(g_con_acceso["t_min"], weights=g_con_acceso["peso"])
            if peso_con_acceso > 0 else np.nan
        )
        peso_total = peso_con_acceso + peso_sin_acceso
        filas.append({
            "nivel_id": nivel_id,
            "nivel_nombre": g["nivel_nombre"].iloc[0],
            "t_min_medio_ponderado": t_medio,
            "poblacion_con_acceso": float(peso_con_acceso),
            "poblacion_sin_acceso_enrutable": float(peso_sin_acceso),
            "pct_poblacion_con_acceso": round(100 * peso_con_acceso / peso_total, 2) if peso_total else 0.0,
            "n_puntos": len(g),
        })

    reporte = pd.DataFrame(filas).sort_values("t_min_medio_ponderado", ascending=False, na_position="last")

    peso_global = con_acceso["peso"].sum() if len(con_acceso) else 0.0
    t_medio_global = np.average(con_acceso["t_min"], weights=con_acceso["peso"]) if peso_global > 0 else float("nan")
    logger.info(
        "weighted_mean_access(level=%s): %d unidades geográficas con al menos 1 punto muestreado "
        "(de las que existen en el universo de demanda_clean). t_min medio ponderado global: %.1f min "
        "(sobre %.0f de población con acceso enrutable conocido).",
        level, len(reporte), t_medio_global, peso_global,
    )
    return reporte.reset_index(drop=True)


# ---------------------------------------------------------------------------
# critical_gap_ranking
# ---------------------------------------------------------------------------

def critical_gap_ranking(
    weighted_mean_access_df: pd.DataFrame, n: Optional[int] = None, poblacion_minima: Optional[float] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Top-n unidades geográficas con peor (mayor) t_min_medio_ponderado.

    poblacion_minima filtra unidades con población insignificante (ver
    config.md > metricas.ranking_criticos.poblacion_minima) ANTES de
    rankear -- sin esto, un centro poblado de pocos habitantes con un
    t_min alto puede dominar el ranking sin ser un caso poblacionalmente
    relevante. El filtro se documenta explícitamente en la columna
    'poblacion_minima_aplicada' del resultado, no se aplica en silencio."""
    cfg = cfg or load_config()
    n = n if n is not None else cfg["metricas"]["ranking_criticos"]["n"]
    poblacion_minima = (
        poblacion_minima if poblacion_minima is not None else cfg["metricas"]["ranking_criticos"]["poblacion_minima"]
    )

    df = weighted_mean_access_df.dropna(subset=["t_min_medio_ponderado"]).copy()
    n_antes = len(df)
    df = df[df["poblacion_con_acceso"] >= poblacion_minima]
    n_filtrado = n_antes - len(df)

    ranking = df.sort_values("t_min_medio_ponderado", ascending=False).head(n).reset_index(drop=True)
    ranking.insert(0, "ranking", np.arange(1, len(ranking) + 1))
    ranking["poblacion_minima_aplicada"] = poblacion_minima

    logger.info(
        "critical_gap_ranking: top-%d de %d unidades (de %d, %d excluidas por poblacion_con_acceso < %.0f). "
        "Peor caso: %s (%.1f min).",
        n, len(df), n_antes, n_filtrado, poblacion_minima,
        ranking["nivel_nombre"].iloc[0] if len(ranking) else "N/A",
        ranking["t_min_medio_ponderado"].iloc[0] if len(ranking) else float("nan"),
    )
    return ranking


# ---------------------------------------------------------------------------
# gini_access
# ---------------------------------------------------------------------------

def gini_access(access_df: pd.DataFrame, population_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Índice de Gini ponderado por población sobre la distribución de
    t_min (mayor Gini = acceso más desigualmente distribuido entre la
    población, no un juicio de "bueno/malo" per se -- dos distritos pueden
    tener el mismo t_min medio y Gini muy distinto si uno concentra la
    demora en una minoría y el otro la reparte parejo).

    Solo considera puntos con t_min conocido y población > 0 (un punto sin
    ruta no tiene un "tiempo" que ubicar en la curva de Lorenz; se reporta
    cuántos quedaron fuera, ver 'n_excluidos_sin_acceso' en el resumen).

    Devuelve (resumen, curva_lorenz):
        resumen: 1 fila [gini, n_puntos, poblacion_total_considerada, n_excluidos_sin_acceso]
        curva_lorenz: [pct_poblacion_acumulada, pct_tiempo_acumulado] -- para graficar en Fase 4.
    """
    merged = _join_by_id(access_df, population_df)
    merged = merged.assign(peso=_effective_weight(merged))

    n_excluidos = int(merged["t_min"].isna().sum())
    valid = merged.dropna(subset=["t_min"])
    valid = valid[valid["peso"] > 0]

    x = valid["t_min"].to_numpy(dtype=float)
    w = valid["peso"].to_numpy(dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]

    W = w.sum()
    wx = w * x
    X = wx.sum()

    if W == 0 or X == 0 or len(x) == 0:
        logger.warning("gini_access: no hay población/puntos válidos para calcular Gini (W=%.2f, X=%.2f).", W, X)
        gini = np.nan
        F = np.array([0.0, 1.0])
        L = np.array([0.0, 1.0]) if X == 0 else np.array([0.0, 0.0])
    else:
        F = np.cumsum(w) / W
        L = np.cumsum(wx) / X
        F_prev = np.concatenate([[0.0], F[:-1]])
        L_prev = np.concatenate([[0.0], L[:-1]])
        gini = float(1 - np.sum((L + L_prev) * (F - F_prev)))
        F = np.concatenate([[0.0], F])
        L = np.concatenate([[0.0], L])

    resumen = pd.DataFrame([{
        "gini": gini,
        "n_puntos": len(x),
        "poblacion_total_considerada": float(W),
        "n_excluidos_sin_acceso": n_excluidos,
    }])
    curva_lorenz = pd.DataFrame({"pct_poblacion_acumulada": F, "pct_tiempo_acumulado": L})

    logger.info(
        "gini_access: Gini=%.3f sobre %d puntos (población ponderada total %.0f); "
        "%d puntos excluidos por no tener acceso enrutable.",
        gini, len(x), W, n_excluidos,
    )
    return resumen, curva_lorenz


# ---------------------------------------------------------------------------
# urban_rural_contrast
# ---------------------------------------------------------------------------

def urban_rural_contrast(access_df: pd.DataFrame, urban_rural_df: pd.DataFrame) -> pd.DataFrame:
    """Contraste de acceso entre 'urbano' y 'rural' (ver classify_urban_rural).

    urban_rural_df debe ser la salida de classify_urban_rural(population_df)
    -- necesita traer 'poblacion_censada' (y opcionalmente 'peso_muestral')
    además de 'zona', porque el contraste es ponderado por población, no un
    promedio simple entre puntos."""
    if "zona" not in urban_rural_df.columns:
        raise KeyError(
            "urban_rural_contrast: urban_rural_df debe venir de classify_urban_rural() "
            "(falta la columna 'zona')."
        )

    merged = _join_by_id(access_df, urban_rural_df)
    merged = merged.assign(peso=_effective_weight(merged))

    filas = []
    for zona, g in merged.groupby("zona"):
        g_con_acceso = g.dropna(subset=["t_min"])
        peso_con_acceso = g_con_acceso["peso"].sum()
        peso_sin_acceso = g.loc[g["flag_sin_acceso_enrutable"].fillna(False), "peso"].sum()
        t_medio = np.average(g_con_acceso["t_min"], weights=g_con_acceso["peso"]) if peso_con_acceso > 0 else np.nan
        filas.append({
            "zona": zona,
            "t_min_medio_ponderado": t_medio,
            "poblacion_con_acceso": float(peso_con_acceso),
            "poblacion_sin_acceso_enrutable": float(peso_sin_acceso),
            "n_puntos": len(g),
        })

    reporte = pd.DataFrame(filas)
    if {"urbano", "rural"}.issubset(set(reporte["zona"])):
        t_urbano = reporte.loc[reporte["zona"] == "urbano", "t_min_medio_ponderado"].iloc[0]
        t_rural = reporte.loc[reporte["zona"] == "rural", "t_min_medio_ponderado"].iloc[0]
        logger.info(
            "urban_rural_contrast: t_min medio urbano=%.1f min vs. rural=%.1f min "
            "(brecha rural/urbano: %.2fx).",
            t_urbano, t_rural, (t_rural / t_urbano) if t_urbano else float("nan"),
        )
    return reporte


# ---------------------------------------------------------------------------
# access_vs_altitude
# ---------------------------------------------------------------------------

def access_vs_altitude(access_df: pd.DataFrame, demanda_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cruce t_min vs. altitud (columna Z de demanda_clean, msnm).

    Devuelve (por_punto, resumen):
        por_punto: [id_demanda, Z, t_min] -- para graficar el scatter en Fase 4.
        resumen: 1 fila [spearman_rho, p_value, n, nota_causalidad]

    Se usa correlación de Spearman (no Pearson) porque no hay razón a priori
    para esperar una relación LINEAL entre altitud y tiempo de acceso -- la
    relación, si existe, viene mediada por terreno/densidad vial, que
    típicamente es monótona pero no lineal (a más altitud, terreno más
    accidentado, menos vías pavimentadas -> mayor t_min, pero no en
    proporción constante).

    NOTA DE CAUSALIDAD (obligatoria, no se omite en el resumen exportado):
    esta es una correlación observacional entre DOS EFECTOS de una causa
    común (la orografía andina/amazónica), no evidencia de que la altitud
    EN SÍ MISMA cause el tiempo de acceso. La altitud está confundida con:
    densidad de red vial, distancia a la instalación resolutiva más
    cercana, y dispersión poblacional -- todas correlacionadas con relieve.
    No se puede, con estos datos, separar "efecto de la altitud" de "efecto
    de estar en zona rural dispersa" (ver urban_rural_contrast). Tratar
    como asociación a describir, NUNCA como relación causal en el informe.
    """
    merged = _join_by_id(access_df, demanda_clean)
    por_punto = merged[["id_demanda", "Z", "t_min"]].copy()

    valid = por_punto.dropna(subset=["Z", "t_min"])
    if len(valid) >= 3:
        rho, pvalue = scipy_stats.spearmanr(valid["Z"], valid["t_min"])
    else:
        rho, pvalue = np.nan, np.nan

    nota = (
        "Correlación observacional, NO causal: altitud y t_min son ambos efectos de la "
        "orografía (relieve -> densidad de red vial -> tiempo de viaje); no se puede aislar "
        "un efecto causal de la altitud en sí con estos datos. Ver docstring de "
        "access_vs_altitude() para el detalle de los confusores (densidad vial, dispersión "
        "poblacional, distancia a instalación resolutiva)."
    )
    resumen = pd.DataFrame([{
        "spearman_rho": rho,
        "p_value": pvalue,
        "n": len(valid),
        "nota_causalidad": nota,
    }])

    logger.info(
        "access_vs_altitude: Spearman rho=%.3f (p=%.4f, n=%d) entre altitud (Z) y t_min. %s",
        rho if not np.isnan(rho) else float("nan"), pvalue if not np.isnan(pvalue) else float("nan"),
        len(valid), "Relación observacional -- ver nota de causalidad en el resumen exportado.",
    )
    return por_punto, resumen


# ---------------------------------------------------------------------------
# Indicadores puntuales para el encabezado del panel (Fase 4)
# ---------------------------------------------------------------------------

def weighted_median_access(access_df: pd.DataFrame, population_df: pd.DataFrame) -> float:
    """Mediana del tiempo de acceso PONDERADA por población.

    Se reporta junto a la media (weighted_mean_access) porque en esta
    distribución las dos cuentan historias distintas y ambas son ciertas: la
    media global sale por encima de las dos horas arrastrada por la cola
    amazónica, mientras que la mediana --- el tiempo de la persona que está
    justo en el medio --- es de decenas de minutos. Un titular que use solo
    una de las dos miente por omisión.

    Solo entran puntos con t_min conocido y peso positivo; devuelve NaN si no
    queda ninguno (p.ej. un filtro del panel sin población censada)."""
    merged = _join_by_id(access_df, population_df)
    merged = merged.assign(peso=_effective_weight(merged))
    valid = merged.dropna(subset=["t_min"])
    valid = valid[valid["peso"] > 0].sort_values("t_min")

    if valid.empty:
        return float("nan")

    peso_acumulado = valid["peso"].cumsum()
    mitad = valid["peso"].sum() / 2.0
    # searchsorted sobre el acumulado: el primer punto que deja al menos la
    # mitad de la población por debajo. No se interpola entre puntos vecinos
    # porque el dato subyacente es discreto (un centro poblado, un tiempo).
    idx = int(np.searchsorted(peso_acumulado.to_numpy(), mitad, side="left"))
    idx = min(idx, len(valid) - 1)
    return float(valid["t_min"].iloc[idx])


def population_within(
    access_df: pd.DataFrame, population_df: pd.DataFrame, umbral_min: float
) -> dict:
    """Población por debajo y por encima de un umbral de minutos.

    Devuelve un dict con la población ponderada bajo el umbral, sobre el
    umbral, sin ruta enrutable y el total considerado, además de los
    porcentajes. Es la función que alimenta el encabezado de indicadores del
    panel y el cálculo de ganancia marginal del simulador de escenarios: los
    dos números tienen que salir de la misma definición o la ganancia no
    cuadraría con la cobertura mostrada.

    La población sin ruta NO se cuenta como ``por encima del umbral'': es
    desconocida, no lenta. Se reporta aparte, igual que en coverage_bands."""
    merged = _join_by_id(access_df, population_df)
    merged = merged.assign(peso=_effective_weight(merged))

    sin_ruta = float(merged.loc[merged["t_min"].isna(), "peso"].sum())
    con_ruta = merged.dropna(subset=["t_min"])
    bajo = float(con_ruta.loc[con_ruta["t_min"] <= umbral_min, "peso"].sum())
    sobre = float(con_ruta.loc[con_ruta["t_min"] > umbral_min, "peso"].sum())
    total = bajo + sobre + sin_ruta

    return {
        "umbral_min": float(umbral_min),
        "poblacion_total": total,
        "poblacion_bajo_umbral": bajo,
        "poblacion_sobre_umbral": sobre,
        "poblacion_sin_ruta": sin_ruta,
        "pct_bajo_umbral": round(100 * bajo / total, 2) if total else 0.0,
        "pct_sobre_umbral": round(100 * sobre / total, 2) if total else 0.0,
        "pct_sin_ruta": round(100 * sin_ruta / total, 2) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Simulador de escenarios (Fase 4)
# ---------------------------------------------------------------------------

def apply_upgrades(
    access_df: pd.DataFrame, improvements_df: pd.DataFrame, upgraded_ids: Iterable
) -> pd.DataFrame:
    """Recalcula t_min suponiendo que los establecimientos de `upgraded_ids`
    pasan a tener capacidad resolutiva.

    `improvements_df` es la matriz de pares (id_demanda, id_candidato,
    t_candidato_min) de Fase 4: para cada centro poblado, el tiempo hacia
    cada candidato a ascenso. El escenario nuevo es, por definición del
    indicador, el mínimo entre el tiempo actual y el tiempo al mejor
    candidato ascendido:

        t_nuevo(i) = min( t_actual(i), min_{j in ascendidos} t(i, j) )

    Un punto que hoy no tiene ninguna ruta (t_min NaN) SÍ puede pasar a
    tenerla si un candidato alcanzable es ascendido; por eso el mínimo se
    calcula tratando el NaN como infinito y no se descarta esa fila.

    Devuelve una copia de access_df con `t_min` actualizado,
    `flag_sin_acceso_enrutable` recalculado y dos columnas nuevas:
    `t_min_original` y `mejorado` (bool), para que el panel pueda mostrar
    quién ganó qué en vez de solo el agregado."""
    upgraded_ids = set(upgraded_ids)
    out = access_df.copy()
    out["t_min_original"] = out["t_min"]

    if not upgraded_ids or improvements_df.empty:
        out["mejorado"] = False
        return out

    relevantes = improvements_df[improvements_df["id_candidato"].isin(upgraded_ids)]
    if relevantes.empty:
        out["mejorado"] = False
        logger.info("apply_upgrades: %d candidatos seleccionados, ninguno mejora a ningún punto "
                    "del subconjunto actual.", len(upgraded_ids))
        return out

    mejor = relevantes.groupby("id_demanda")["t_candidato_min"].min()
    t_candidato = out["id_demanda"].map(mejor)

    # np.fmin ignora el NaN del lado que lo tenga (a diferencia de np.minimum,
    # que propaga NaN): exactamente el comportamiento que se necesita para que
    # un punto hoy sin ruta pueda pasar a tenerla.
    out["t_min"] = np.fmin(out["t_min"].to_numpy(dtype=float),
                            t_candidato.to_numpy(dtype=float))
    out["mejorado"] = out["t_min"] < out["t_min_original"].fillna(np.inf)
    out["flag_sin_acceso_enrutable"] = out["t_min"].isna()

    logger.info(
        "apply_upgrades: %d establecimientos ascendidos mejoran el tiempo de %d de %d puntos "
        "de demanda del subconjunto.",
        len(upgraded_ids), int(out["mejorado"].sum()), len(out),
    )
    return out


def ranking_upgrade_gain(
    access_df: pd.DataFrame, population_df: pd.DataFrame, improvements_df: pd.DataFrame,
    umbral_min: float, top_n: int = 20,
) -> pd.DataFrame:
    """Ganancia INDIVIDUAL de ascender cada establecimiento candidato: cuánta
    población pasaría a estar dentro del umbral si se ascendiera ese
    establecimiento y ningún otro.

    Sirve para que el usuario del simulador no tenga que adivinar cuáles de
    los 607 candidatos vale la pena mirar. Es explícitamente una ganancia
    individual, NO el reparto de un total: las ganancias de dos candidatos
    vecinos se solapan (la misma población entra dentro del umbral con
    cualquiera de los dos), así que sumarlas sobreestima el efecto conjunto.
    La ganancia real de una combinación se obtiene con apply_upgrades sobre
    esa combinación --- que es justamente lo que hace el simulador cuando el
    usuario selecciona más de uno.

    Devuelve una fila por candidato con ganancia positiva, ordenada de mayor
    a menor: [id_candidato, poblacion_ganada, n_puntos_ganados]."""
    merged = _join_by_id(access_df, population_df)
    merged = merged.assign(peso=_effective_weight(merged))
    estado = merged[["id_demanda", "t_min", "peso"]]

    pares = improvements_df.merge(estado, on="id_demanda", how="inner")
    # Fuera del umbral hoy (o sin ruta), dentro del umbral con el ascenso.
    fuera_hoy = pares["t_min"].isna() | (pares["t_min"] > umbral_min)
    dentro_manana = pares["t_candidato_min"] <= umbral_min
    ganan = pares[fuera_hoy & dentro_manana & (pares["peso"] > 0)]

    if ganan.empty:
        return pd.DataFrame(columns=["id_candidato", "poblacion_ganada", "n_puntos_ganados"])

    ranking = (
        ganan.groupby("id_candidato")
        .agg(poblacion_ganada=("peso", "sum"), n_puntos_ganados=("id_demanda", "nunique"))
        .reset_index()
        .sort_values("poblacion_ganada", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info(
        "ranking_upgrade_gain(umbral=%.0f min): %d candidatos con ganancia positiva; "
        "el mejor individual acercaría a %.0f habitantes.",
        umbral_min, ganan["id_candidato"].nunique(),
        ranking["poblacion_ganada"].iloc[0] if len(ranking) else 0.0,
    )
    return ranking
