"""
population.py — Fase 3: población por centro poblado
========================================================

demanda_clean.parquet trae CPINEI (código INEI de centro poblado, 10
dígitos: ubigeo_distrito[6] + correlativo_cp[4]) pero NO población -- el
shapefile de SIGMED del que sale demanda_clean no la incluye. Este módulo:

  1. Descarga la tabla oficial de población por centro poblado del Censo
     INEI 2017 ("Directorio Nacional de Centros Poblados", Lib1541), un
     .xlsx por departamento, y la parsea a un DataFrame plano
     [CPINEI, nombre_centro_poblado, poblacion_censada].
  2. La une a demanda_clean por CPINEI.

Filosofía (misma que validation.py): un centro poblado que no matchea NO se
descarta en silencio. Se conserva con `flag_sin_poblacion=True` y
`poblacion_censada=NaN`, y se cuenta/reporta explícitamente -- porque
"cuántos puntos de demanda quedan sin población" es en sí mismo un hallazgo
relevante para el informe (ver justificación en `merge_population_to_demand`).

Fuente: ver config.md > fuentes.censo_poblacion_centros_poblados. El portal
no expone una API ni un CSV único; solo un .xlsx jerárquico por departamento
(depto -> provincia -> distrito -> centro poblado). Por eso este módulo
sigue el mismo patrón de acquisition.py (descarga cacheada, re-ejecutable,
nunca falla en silencio) en vez de asumir un formato tabular simple.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.acquisition import (
    AcquisitionError,
    _already_downloaded,
    _download_stream,
    _stamp_download_date,
)
from src.config_loader import REPO_ROOT, load_config, path_for

logger = logging.getLogger("population")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # NO usamos logging.basicConfig() aquí: acquisition.py (importado arriba)
    # ya lo llama con "acquisition:" hardcodeado en el formato en vez de
    # %(name)s, y basicConfig() solo tiene efecto en la primera llamada del
    # proceso -- si population.py corriera después, todos los logs de este
    # módulo saldrían etiquetados "acquisition:". Un handler propio evita
    # heredar ese bug preexistente sin tocar los módulos de Fase 1/2.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] population: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# Tolerancia para la verificación de coherencia entre la suma de población
# de centros poblados parseados y el total departamental que la propia
# planilla del INEI reporta en su fila de cabecera (ver _parse_departamento_rows).
_TOLERANCIA_COHERENCIA_DEPARTAMENTO = 0.005  # 0.5%


# ---------------------------------------------------------------------------
# 1. Descarga (un .xlsx por departamento configurado en config.md)
# ---------------------------------------------------------------------------

def download_poblacion_centros_poblados(cfg: Optional[dict] = None, force: bool = False) -> dict[str, Path]:
    """Descarga el .xlsx de población censal por centro poblado para cada
    departamento en cfg['departamentos'] (costero/andino/amazonico).

    Devuelve {ubigeo_dep: Path}. Si una descarga falla y no hay caché local,
    se levanta AcquisitionError con instrucción de descarga manual -- mismo
    contrato que acquisition.download_admin_boundaries.
    """
    cfg = cfg or load_config()
    fuente = cfg["fuentes"]["censo_poblacion_centros_poblados"]

    raw_dir = REPO_ROOT / cfg["rutas"]["poblacion_cp_raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    rutas: dict[str, Path] = {}
    for region, dep_cfg in cfg["departamentos"].items():
        ubigeo_dep = dep_cfg["ubigeo_dep"]
        dest = raw_dir / f"dpto{ubigeo_dep}.xlsx"

        if _already_downloaded(dest) and not force:
            logger.info("Población %s (%s, ubigeo_dep=%s) ya existe en caché (%s).",
                        region, dep_cfg["nombre"], ubigeo_dep, dest)
            rutas[ubigeo_dep] = dest
            continue

        url = fuente["url_template_departamento"].format(ubigeo_dep=ubigeo_dep)
        try:
            _download_stream(url, dest)
            _stamp_download_date(f"fuentes.censo_poblacion_centros_poblados.dpto{ubigeo_dep}")
            rutas[ubigeo_dep] = dest
        except Exception as e:
            logger.warning("Descarga de población para %s (dpto%s) falló: %s", dep_cfg["nombre"], ubigeo_dep, e)
            if _already_downloaded(dest):
                logger.warning("Usando copia cacheada existente (posiblemente desactualizada).")
                rutas[ubigeo_dep] = dest
                continue
            raise AcquisitionError(
                f"No se pudo descargar la población censal de {dep_cfg['nombre']} (dpto{ubigeo_dep}) "
                f"desde {url}, y no hay caché local. Descarga manual: ir a {fuente['portal']}, "
                f"bajar el departamento correspondiente y guardarlo en {dest}."
            ) from e

    return rutas


# ---------------------------------------------------------------------------
# 2. Parseo del .xlsx jerárquico (depto -> provincia -> distrito -> CP)
# ---------------------------------------------------------------------------

def _parse_poblacion_value(v: object) -> float:
    """Convierte una celda de 'POBLACIÓN CENSADA > Total' a float.
    '-' significa 0 (centro poblado con viviendas pero sin personas
    presentes al momento del censo -- ver nota 1/ de la planilla del INEI,
    no es un dato faltante)."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "-":
        return 0.0
    if s == "":
        return np.nan
    s = s.replace(" ", "").replace("\xa0", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_departamento_rows(rows: list[tuple], ubigeo_dep: str) -> tuple[pd.DataFrame, Optional[float]]:
    """Parsea las filas crudas (values_only) de la hoja de un departamento.

    La planilla del INEI es una tabla jerárquica de ancho fijo, NO un CSV
    tabular limpio: cada fila de "detalle" puede ser un departamento, una
    provincia, un distrito o un centro poblado, distinguibles solo por el
    prefijo de texto en la columna 'CENTROS POBLADOS' (columna B). Se recorre
    linealmente llevando el ubigeo del distrito vigente, que es el prefijo
    (6 dígitos) del CPINEI de cada centro poblado que sigue.

    Devuelve (df_centros_poblados, poblacion_total_departamento) -- el
    segundo valor viene de la fila 'DEPARTAMENTO ...' y sirve solo para la
    verificación de coherencia en load_poblacion_departamento_xlsx.
    """
    registros = []
    ubigeo_distrito_actual: Optional[str] = None
    poblacion_total_departamento: Optional[float] = None

    for row in rows:
        codigo = row[0] if len(row) > 0 else None
        nombre = row[1] if len(row) > 1 else None
        poblacion_raw = row[4] if len(row) > 4 else None

        if codigo is None or nombre is None:
            continue
        codigo = str(codigo).strip()
        nombre_str = str(nombre).strip()
        if codigo == "" or nombre_str == "":
            continue

        if nombre_str.startswith("DEPARTAMENTO"):
            poblacion_total_departamento = _parse_poblacion_value(poblacion_raw)
            continue
        if nombre_str.startswith("PROVINCIA"):
            continue
        if nombre_str.startswith("DISTRITO"):
            ubigeo_distrito_actual = codigo.zfill(6)
            continue

        # Fila de centro poblado: solo es interpretable si ya vimos su
        # distrito contenedor. Si no (nota al pie, encabezado suelto), se
        # ignora -- no es un centro poblado real.
        if ubigeo_distrito_actual is None:
            continue

        cpinei = ubigeo_distrito_actual + codigo.zfill(4)
        registros.append({
            "CPINEI": cpinei,
            "ubigeo_dep": ubigeo_dep,
            "ubigeo_distrito": ubigeo_distrito_actual,
            "nombre_centro_poblado": nombre_str,
            "poblacion_censada": _parse_poblacion_value(poblacion_raw),
        })

    return pd.DataFrame(registros), poblacion_total_departamento


def load_poblacion_departamento_xlsx(path: Path, ubigeo_dep: str) -> pd.DataFrame:
    """Abre el .xlsx de un departamento y devuelve el DataFrame de centros
    poblados, verificando coherencia contra el total departamental reportado
    en la propia planilla (no descarta nada si no coincide -- solo advierte,
    porque una fuente de datos administrativa peruana con leves
    inconsistencias internas es un hallazgo a documentar, no un bug a
    esconder)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheet_name = ubigeo_dep if ubigeo_dep in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    df, poblacion_total_reportada = _parse_departamento_rows(rows, ubigeo_dep)

    if poblacion_total_reportada is not None and len(df):
        suma = df["poblacion_censada"].sum()
        if poblacion_total_reportada > 0:
            desvio = abs(suma - poblacion_total_reportada) / poblacion_total_reportada
            if desvio > _TOLERANCIA_COHERENCIA_DEPARTAMENTO:
                logger.warning(
                    "dpto%s: suma de población de centros poblados parseados (%.0f) difiere del "
                    "total departamental reportado en la planilla (%.0f) en %.2f%% (> tolerancia %.1f%%). "
                    "Revisar manualmente el parseo de %s.",
                    ubigeo_dep, suma, poblacion_total_reportada, 100 * desvio,
                    100 * _TOLERANCIA_COHERENCIA_DEPARTAMENTO, path,
                )

    logger.info("dpto%s: %d centros poblados parseados desde %s (población total: %.0f).",
                ubigeo_dep, len(df), path.name, df["poblacion_censada"].sum() if len(df) else 0)
    return df


# ---------------------------------------------------------------------------
# 3. Orquestador: descarga + parseo + caché de la tabla de población
# ---------------------------------------------------------------------------

def build_poblacion_centros_poblados(cfg: Optional[dict] = None, force: bool = False) -> pd.DataFrame:
    """Devuelve la tabla completa de población por centro poblado para los 3
    departamentos del proyecto. Cacheada en Parquet (mismo patrón que
    compute_od_matrix): si el archivo procesado ya existe, se reutiliza sin
    volver a descargar/parsear -- borra el archivo para forzar recálculo."""
    cfg = cfg or load_config()
    cache_path = path_for("poblacion_cp_processed", cfg)

    if cache_path.exists() and not force:
        logger.info("Tabla de población ya cacheada en %s, se reutiliza (borra el archivo para recalcular).",
                    cache_path)
        return pd.read_parquet(cache_path)

    rutas = download_poblacion_centros_poblados(cfg, force=force)
    dfs = [load_poblacion_departamento_xlsx(path, ubigeo_dep) for ubigeo_dep, path in rutas.items()]
    poblacion = pd.concat(dfs, ignore_index=True)

    n_dup = poblacion["CPINEI"].duplicated().sum()
    if n_dup:
        logger.warning(
            "%d CPINEI duplicados entre departamentos descargados (inesperado: los ubigeo de "
            "distrito no deberían repetirse entre departamentos distintos). Se conserva la "
            "primera ocurrencia; revisar manualmente si esto persiste.",
            n_dup,
        )
        poblacion = poblacion.drop_duplicates(subset="CPINEI", keep="first")

    poblacion.to_parquet(cache_path, index=False)
    logger.info(
        "Tabla de población escrita en caché: %s (%d centros poblados, %d departamentos, población total %.0f).",
        cache_path, len(poblacion), poblacion["ubigeo_dep"].nunique(), poblacion["poblacion_censada"].sum(),
    )
    return poblacion


# ---------------------------------------------------------------------------
# 4. Unión a demanda_clean
# ---------------------------------------------------------------------------

def merge_population_to_demand(
    demanda_df: pd.DataFrame,
    poblacion_df: Optional[pd.DataFrame] = None,
    *,
    cfg: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une población a demanda_clean por CPINEI. Devuelve (demanda_con_poblacion, reporte).

    Ningún registro de demanda_clean se descarta: un CPINEI que no matchea
    (o que está ausente) queda con poblacion_censada=NaN y flag_sin_poblacion=True,
    igual que el patrón flag_* de validation.py.

    Dos causas distintas de "sin población", reportadas por separado porque
    tienen implicancias distintas para el informe:
      - flag_sin_cpinei: el centro poblado no viene del Censo 2017 (su
        FUENTE_INE es otra: INEI13, IGN, MED-GPS, etc.) -- no hay forma de
        buscarle población en esta fuente, es una limitación de cobertura
        de la fuente, no un error de join.
      - flag_cpinei_sin_poblacion: el registro SÍ trae un CPINEI con formato
        de Censo 2017 pero no aparece en la tabla de población descargada
        (posible error de digitación, centro poblado dado de baja/fusionado
        entre 2017 y la fecha del shapefile de SIGMED, etc.) -- esto sí
        amerita revisión manual si el conteo es alto.
    """
    cfg = cfg or load_config()
    if poblacion_df is None:
        poblacion_df = build_poblacion_centros_poblados(cfg)

    poblacion_lookup = poblacion_df[["CPINEI", "nombre_centro_poblado", "poblacion_censada"]].rename(
        columns={"nombre_centro_poblado": "nombre_centro_poblado_censo2017"}
    )

    merged = demanda_df.merge(poblacion_lookup, on="CPINEI", how="left", validate="m:1")
    flag_sin_cpinei = merged["CPINEI"].isna()
    flag_cpinei_sin_poblacion = merged["CPINEI"].notna() & merged["poblacion_censada"].isna()
    # .assign() en vez de asignación incremental: pandas 2.2 emite un falso
    # positivo "ChainedAssignmentError" al setear columnas una a una sobre un
    # DataFrame recién salido de merge(), incluso con copy_on_write=False.
    out = merged.assign(
        flag_sin_cpinei=flag_sin_cpinei,
        flag_cpinei_sin_poblacion=flag_cpinei_sin_poblacion,
        flag_sin_poblacion=flag_sin_cpinei | flag_cpinei_sin_poblacion,
    )

    n_total = len(out)
    n_sin_cpinei = int(out["flag_sin_cpinei"].sum())
    n_cpinei_sin_match = int(out["flag_cpinei_sin_poblacion"].sum())
    n_sin_poblacion = int(out["flag_sin_poblacion"].sum())

    logger.info(
        "merge_population_to_demand: %d puntos de demanda -> %d (%.1f%%) sin CPINEI, "
        "%d (%.1f%%) con CPINEI pero sin match en el censo, %d (%.1f%%) sin población en total. "
        "Ninguno se descarta; quedan con poblacion_censada=NaN para auditoría.",
        n_total, n_sin_cpinei, 100 * n_sin_cpinei / n_total if n_total else 0,
        n_cpinei_sin_match, 100 * n_cpinei_sin_match / n_total if n_total else 0,
        n_sin_poblacion, 100 * n_sin_poblacion / n_total if n_total else 0,
    )

    reporte = pd.DataFrame([
        {
            "regla": "sin_cpinei",
            "n_marcados": n_sin_cpinei,
            "pct_del_total": round(100 * n_sin_cpinei / n_total, 2) if n_total else 0.0,
            "accion": "mantenido con advertencia (poblacion_censada=NaN, excluido de agregados ponderados por población)",
            "justificacion": "El centro poblado no proviene del Censo 2017 (FUENTE_INE distinta de INEI17); "
                              "esta fuente de población no tiene forma de cubrirlo.",
        },
        {
            "regla": "cpinei_sin_match_en_censo",
            "n_marcados": n_cpinei_sin_match,
            "pct_del_total": round(100 * n_cpinei_sin_match / n_total, 2) if n_total else 0.0,
            "accion": "mantenido con advertencia (poblacion_censada=NaN, excluido de agregados ponderados por población)",
            "justificacion": "Trae CPINEI con formato de Censo 2017 pero no aparece en la tabla de "
                              "población descargada; revisar manualmente si el conteo es alto "
                              "(posible centro poblado dado de baja/fusionado, o error de digitación).",
        },
    ])

    reporte_path = path_for("poblacion_match_report", cfg)
    reporte.to_csv(reporte_path, index=False)
    logger.info("Reporte de matching de población escrito en %s", reporte_path)

    return out, reporte


if __name__ == "__main__":
    cfg = load_config()
    demanda = pd.read_parquet(path_for("demanda_processed", cfg))
    poblacion = build_poblacion_centros_poblados(cfg)
    demanda_con_poblacion, reporte = merge_population_to_demand(demanda, poblacion, cfg=cfg)

    demanda_con_poblacion.to_parquet(path_for("demanda_con_poblacion", cfg), index=False)

    print("\n--- REPORTE DE MATCHING DE POBLACIÓN ---")
    print(reporte.to_string(index=False))
    print(f"\nColumnas de demanda_con_poblacion ({len(demanda_con_poblacion.columns)}):")
    for c in demanda_con_poblacion.columns:
        print(f"  - {c}  ({demanda_con_poblacion[c].dtype})")
    print(f"\nGuardado: {path_for('demanda_con_poblacion', cfg)}")
