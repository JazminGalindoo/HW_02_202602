"""
config_loader.py
-----------------
Extrae el bloque ```yaml ... ``` de config.md y lo expone como un dict de
Python. Esto permite que config.md sea al mismo tiempo la documentación
legible por humanos que pide la consigna Y la fuente de verdad que consume
el código -- no hay dos copias de los parámetros que puedan desincronizarse.

Uso:
    from src.config_loader import load_config
    cfg = load_config()
    cfg["departamentos"]["costero"]["nombre"]  # "PIURA"
"""

from __future__ import annotations

import re
import functools
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.md"

_YAML_BLOCK_RE = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)


@functools.lru_cache(maxsize=1)
def load_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Lee config.md, extrae el primer bloque ```yaml``` y lo parsea.

    Cacheado con lru_cache: se lee una sola vez por proceso. Si necesitas
    forzar una relectura (p.ej. en tests que modifican config.md), llama a
    load_config.cache_clear() primero.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró config.md en {config_path}. "
            "Este archivo es obligatorio: contiene todos los parámetros del pipeline."
        )

    text = config_path.read_text(encoding="utf-8")
    match = _YAML_BLOCK_RE.search(text)
    if not match:
        raise ValueError(
            "config.md no contiene un bloque ```yaml ... ``` parseable. "
            "Revisa que el bloque de configuración no se haya roto al editar."
        )

    cfg = yaml.safe_load(match.group(1))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    """Chequeos mínimos de forma para fallar rápido y con un mensaje claro,
    en vez de un KeyError críptico tres módulos más adelante."""
    required_top_level = [
        "departamentos", "bbox_peru", "categorias_resolutivas",
        "categorias_no_resolutivas", "normalizacion_categoria",
        "estados_activos", "rutas", "fuentes", "enrutamiento",
    ]
    missing = [k for k in required_top_level if k not in cfg]
    if missing:
        raise ValueError(f"config.md: faltan claves obligatorias en el YAML: {missing}")

    for region in ("costero", "andino", "amazonico"):
        if region not in cfg["departamentos"]:
            raise ValueError(f"config.md: falta el departamento '{region}' en 'departamentos'")

    motor = cfg["enrutamiento"].get("motor")
    if motor not in ("osrm", "osmnx"):
        raise ValueError(f"config.md: enrutamiento.motor debe ser 'osrm' u 'osmnx', encontrado: {motor!r}")


def path_for(key: str, cfg: Dict[str, Any] | None = None) -> Path:
    """Resuelve una entrada de cfg['rutas'][key] a un Path absoluto bajo la
    raíz del repo. Crea los directorios padre si no existen (para outputs)."""
    cfg = cfg or load_config()
    rel = cfg["rutas"][key]
    p = REPO_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
