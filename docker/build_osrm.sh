#!/usr/bin/env bash
# build_osrm.sh
# --------------
# Compila los tres grafos OSRM (car, bike, foot) a partir de
# data/raw/peru-latest.osm.pbf. Cada perfil necesita su propio pipeline
# extract -> partition -> customize (algoritmo MLD), porque el grafo
# resultante depende del perfil (p.ej. autopistas cuentan para car, no
# para foot).
#
# Re-ejecutable: si el .osrm ya existe para un perfil, se salta ese perfil
# a menos que se pase --force.
#
# Uso:
#   bash docker/build_osrm.sh           # construye los 3 perfiles que falten
#   bash docker/build_osrm.sh --force   # reconstruye todo desde cero

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_PBF="$ROOT_DIR/data/raw/peru-latest.osm.pbf"
OSRM_DIR="$ROOT_DIR/data/raw/osrm"
FORCE="${1:-}"

if [ ! -f "$RAW_PBF" ]; then
  echo "ERROR: no se encontró $RAW_PBF"
  echo "Corre primero: python -m src.acquisition"
  exit 1
fi

mkdir -p "$OSRM_DIR"/{car,bike,foot}

build_profile () {
  local profile="$1"
  local osrm_profile_file="$2"   # perfil Lua incluido en la imagen oficial de OSRM
  local out_dir="$OSRM_DIR/$profile"
  local out_pbf="$out_dir/peru-latest.osm.pbf"
  local out_osrm="$out_dir/peru-latest.osrm"

  if [ -f "$out_osrm" ] && [ "$FORCE" != "--force" ]; then
    echo "[$profile] ya compilado en $out_osrm, se omite (usa --force para rehacer)."
    return
  fi

  echo "=== [$profile] extract ==="
  cp -f "$RAW_PBF" "$out_pbf"
  docker run --rm -t -v "$out_dir:/data" osrm/osrm-backend \
    osrm-extract -p "/opt/${osrm_profile_file}" "/data/peru-latest.osm.pbf"

  echo "=== [$profile] partition ==="
  docker run --rm -t -v "$out_dir:/data" osrm/osrm-backend \
    osrm-partition "/data/peru-latest.osrm"

  echo "=== [$profile] customize ==="
  docker run --rm -t -v "$out_dir:/data" osrm/osrm-backend \
    osrm-customize "/data/peru-latest.osrm"

  echo "[$profile] listo."
}

build_profile "car"  "car.lua"
build_profile "bike" "bicycle.lua"
build_profile "foot" "foot.lua"

echo ""
echo "Los 3 grafos están listos en $OSRM_DIR/{car,bike,foot}."
echo "Levanta los servidores con:  docker compose -f docker/docker-compose.yml up -d"
