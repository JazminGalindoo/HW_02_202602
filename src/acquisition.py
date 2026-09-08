"""
acquisition.py — Fase 1: adquisición de datos
==============================================

Descarga las cuatro fuentes requeridas:
    1. RENIPRESS (oferta de salud)          -> data/raw/renipress.csv
    2. Centros poblados SIGMED (demanda)    -> data/raw/centros_poblados.gpkg
    3. Extracto OSM de Perú (red vial)      -> data/raw/peru-latest.osm.pbf
    4. Límites distritales                  -> data/raw/limites_distritales.geojson

Principios de diseño (ver consigna):
    - Los archivos crudos NUNCA se modifican in situ.
    - El paso es re-ejecutable: si el archivo de destino ya existe y su
      tamaño es razonable, no se vuelve a descargar.
    - Si una fuente cae, se documenta la fecha de intento y se usa un
      caché/alternativa si existe, en vez de fallar en silencio.
    - Cada descarga exitosa registra su fecha en config.md (campo
      fecha_descarga_cache) para trazabilidad -- ver `_stamp_download_date`.

Este módulo es importable y cada función se puede probar de forma aislada
(no hay lógica de descarga escondida dentro de un notebook o un script `if
__name__ == "__main__"` monolítico).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import requests

from src.config_loader import load_config, path_for, REPO_ROOT

logger = logging.getLogger("acquisition")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] acquisition: %(message)s",
)

CHUNK = 1024 * 1024  # 1 MB
MIN_VALID_SIZE_BYTES = 2048  # por debajo de esto, asumimos descarga rota/HTML de error


class AcquisitionError(RuntimeError):
    """Fuente inaccesible y no hay caché disponible. El pipeline debe poder
    capturar esto y seguir documentando en vez de reventar sin explicación."""


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def _already_downloaded(dest: Path) -> bool:
    return dest.exists() and dest.stat().st_size > MIN_VALID_SIZE_BYTES


def _download_stream(url: str, dest: Path, *, headers: Optional[dict] = None,
                      timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Descargando %s -> %s", url, dest)
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        if written % (CHUNK * 20) < CHUNK:  # loggea cada ~20MB
                            logger.info("  %.1f%% (%d / %d MB)", pct,
                                        written // (1024 * 1024), total // (1024 * 1024))
    tmp.rename(dest)
    logger.info("OK: %s (%.2f MB)", dest.name, dest.stat().st_size / (1024 * 1024))


def _stamp_download_date(cfg_key_path: str) -> None:
    """Registra en un log JSON aparte (no reescribimos config.md por
    programa para no pisar comentarios/formato humano) la fecha en que cada
    fuente fue efectivamente descargada. Ver logs/download_dates.json."""
    log_path = REPO_ROOT / "logs" / "download_dates.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding="utf-8"))
    data[cfg_key_path] = dt.datetime.now(dt.timezone.utc).isoformat()
    log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. RENIPRESS
# ---------------------------------------------------------------------------

def resolve_renipress_csv_url(cfg: dict) -> str:
    """El portal datosabiertos.gob.pe corre sobre DKAN, que expone una API
    compatible con CKAN (`package_show`). En vez de hardcodear la URL del
    recurso CSV (que trae un UUID que cambia si SUSALUD re-sube el dataset),
    resolvemos la URL vigente en tiempo de ejecución.

    Si la API cambia de forma o el portal está caído, esto lanza
    AcquisitionError con un mensaje claro -- que acquisition.download_all()
    sabe capturar y resolver con el caché local.
    """
    fuente = cfg["fuentes"]["renipress"]
    api_url = fuente["api_package_show"].format(slug=fuente["dataset_slug"])
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise AcquisitionError(f"No se pudo consultar la API de RENIPRESS ({api_url}): {e}")

    resources = payload.get("result", {}).get("resources", [])
    csv_resources = [r for r in resources if str(r.get("format", "")).upper() == "CSV"]
    if not csv_resources:
        raise AcquisitionError(
            "package_show respondió pero no trae ningún recurso CSV. "
            "Revisar manualmente el dataset en el portal."
        )
    # Tomamos el más reciente por 'last_modified' o 'created' si está disponible.
    csv_resources.sort(key=lambda r: r.get("last_modified") or r.get("created") or "", reverse=True)
    return csv_resources[0]["url"]


def download_renipress(cfg: Optional[dict] = None, force: bool = False) -> Path:
    cfg = cfg or load_config()
    dest = path_for("renipress_raw", cfg)

    if _already_downloaded(dest) and not force:
        logger.info("RENIPRESS ya existe en caché (%s), no se re-descarga.", dest)
        return dest

    try:
        url = resolve_renipress_csv_url(cfg)
        _download_stream(url, dest)
        _stamp_download_date("fuentes.renipress")
        return dest
    except AcquisitionError as e:
        logger.warning("Fallo al resolver/descargar RENIPRESS vía API: %s", e)
        if _already_downloaded(dest):
            logger.warning("Usando copia cacheada existente de RENIPRESS (posiblemente desactualizada).")
            return dest
        raise AcquisitionError(
            "RENIPRESS no está disponible ni por API ni en caché local. "
            "Descarga manual: ir a "
            f"{cfg['fuentes']['renipress']['portal']}/dataset/{cfg['fuentes']['renipress']['dataset_slug']} "
            f"y guardar el CSV en {dest}."
        ) from e


# ---------------------------------------------------------------------------
# 2. Centros poblados (SIGMED) — sin API pública estable, requiere paso manual
# ---------------------------------------------------------------------------

def download_centros_poblados(cfg: Optional[dict] = None, force: bool = False) -> Path:
    """SIGMED no expone un endpoint de descarga programático estable (es un
    visor/descarga espacial interactivo). Este módulo documenta el
    requisito y valida que el archivo exista, en vez de fingir una descarga
    automática que en la práctica no funciona igual dos meses seguidos.

    Procedimiento (una sola vez, documentado en README.md):
        1. Entrar a https://sigmed.minedu.gob.pe/descargas/
        2. Descargar la capa de "Centros poblados" en formato shapefile/gpkg.
        3. Copiar el archivo a data/raw/centros_poblados.gpkg
    """
    cfg = cfg or load_config()
    dest = path_for("centros_poblados_raw", cfg)

    if _already_downloaded(dest) and not force:
        logger.info("Centros poblados ya existen en caché (%s).", dest)
        return dest

    url_manual = cfg["fuentes"]["sigmed_centros_poblados"].get("url_manual")
    if url_manual:
        try:
            _download_stream(url_manual, dest)
            _stamp_download_date("fuentes.sigmed_centros_poblados")
            return dest
        except Exception as e:
            logger.warning("Descarga automática de SIGMED falló: %s", e)

    raise AcquisitionError(
        "No hay archivo de centros poblados en caché y SIGMED no tiene un "
        "enlace de descarga configurado en config.md (fuentes.sigmed_centros_poblados.url_manual). "
        f"Descarga manualmente desde {cfg['fuentes']['sigmed_centros_poblados']['portal']} "
        f"y guarda el resultado en {dest}. Este es un hallazgo legítimo sobre datos "
        "abiertos peruanos, no un fallo del pipeline -- documentarlo en el informe."
    )


# ---------------------------------------------------------------------------
# 3. Extracto OSM de Perú (Geofabrik) — URL estable, descarga directa
# ---------------------------------------------------------------------------

def download_osm_extract(cfg: Optional[dict] = None, force: bool = False) -> Path:
    cfg = cfg or load_config()
    dest = path_for("osm_pbf_raw", cfg)

    if _already_downloaded(dest) and not force:
        logger.info("Extracto OSM ya existe en caché (%s, %.1f MB).",
                    dest, dest.stat().st_size / (1024 * 1024))
        return dest

    url = cfg["fuentes"]["osm_peru"]["url"]
    _download_stream(url, dest, timeout=600)  # ~220MB, dale tiempo
    _stamp_download_date("fuentes.osm_peru")
    return dest


# ---------------------------------------------------------------------------
# 4. Límites administrativos distritales
# ---------------------------------------------------------------------------

def download_admin_boundaries(cfg: Optional[dict] = None, force: bool = False) -> Path:
    cfg = cfg or load_config()
    dest = path_for("limites_distritales_raw", cfg)

    if _already_downloaded(dest) and not force:
        logger.info("Límites distritales ya existen en caché (%s).", dest)
        return dest

    url = cfg["fuentes"]["limites_administrativos"]["url_distritos"]
    try:
        _download_stream(url, dest)
        _stamp_download_date("fuentes.limites_administrativos")
        return dest
    except Exception as e:
        if _already_downloaded(dest):
            logger.warning("Descarga falló (%s), usando caché existente.", e)
            return dest
        raise AcquisitionError(
            f"No se pudieron descargar los límites distritales desde {url}. "
            f"Fuente alternativa: GEOIDEP / IGN (citar en el informe). Error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def download_all(force: bool = False) -> dict:
    """Ejecuta las 4 descargas, capturando fallos individuales para que una
    fuente caída no tumbe todo el paso de adquisición. Devuelve un resumen
    dict apto para loguear o volcar a data/outputs/."""
    cfg = load_config()
    resultado = {}
    steps = {
        "renipress": download_renipress,
        "centros_poblados": download_centros_poblados,
        "osm_extract": download_osm_extract,
        "limites_administrativos": download_admin_boundaries,
    }
    for nombre, fn in steps.items():
        try:
            path = fn(cfg, force=force)
            resultado[nombre] = {"status": "ok", "path": str(path)}
        except AcquisitionError as e:
            logger.error("Fuente '%s' no disponible: %s", nombre, e)
            resultado[nombre] = {"status": "failed", "error": str(e)}

    summary_path = REPO_ROOT / "logs" / "acquisition_summary.json"
    summary_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resumen de adquisición escrito en %s", summary_path)
    return resultado


if __name__ == "__main__":
    download_all()
