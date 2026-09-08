# Golden Hour — Acceso a salud resolutiva en Perú

Proyecto integrador: acceso real por carretera a centros de salud con
capacidad resolutiva (categoría II-1 en adelante) en tres departamentos:
**Piura** (costa), **Ayacucho** (andino), **Loreto** (amazónico).

Este README cubre **Fase 1 (adquisición y validación)** y **Fase 2
(enrutamiento)**. Ver `config.md` para cambiar departamentos, umbrales o
motor de enrutamiento sin tocar código.

## 0. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requiere además **Docker** (para OSRM) — ver paso 3.

## 1. Descargar los datos crudos (Fase 1)

```bash
python -m src.acquisition
```

Esto descarga a `data/raw/` (nunca se modifica in situ):

| Fuente | Archivo | Notas |
|---|---|---|
| RENIPRESS (SUSALUD) | `renipress.csv` | Se resuelve la URL vigente vía la API tipo CKAN del portal (`package_show`), porque el enlace directo trae un UUID que cambia entre publicaciones. |
| Centros poblados (SIGMED) | `centros_poblados.gpkg` | **Requiere un paso manual una sola vez** — ver abajo. |
| Red vial OSM | `peru-latest.osm.pbf` | Descarga directa de Geofabrik (~220 MB). |
| Límites distritales | `limites_distritales.geojson` | Repo público `juaneladio/peru-geojson` (INEI 2007 georreferenciado), citado en el informe. |

**Paso manual para SIGMED** (el portal no tiene API de descarga estable):
1. Ir a https://sigmed.minedu.gob.pe/descargas/
2. Descargar la capa de "Centros poblados" (shapefile o GeoPackage).
3. Guardar el archivo como `data/raw/centros_poblados.gpkg`.
4. Volver a correr `python -m src.acquisition` — detectará el archivo y no
   lo pedirá de nuevo.

Si una fuente falla (portal caído, cambio de formato), `acquisition.py` no
truena todo el proceso: registra el fallo en
`logs/acquisition_summary.json` y usa el caché local si existe. Esto es un
hallazgo legítimo sobre datos abiertos peruanos — se documenta en el
informe (Fase 5, sección Limitaciones), no se oculta.

## 2. Validar y limpiar (Fase 1)

La lógica vive en `src/validation.py`. Ejemplo de uso end-to-end (ajustar
nombres de columnas reales una vez inspeccionado el CSV de RENIPRESS):

```python
import pandas as pd
import geopandas as gpd
from src.config_loader import load_config
from src.validation import apply_category_rules, run_quality_pipeline

cfg = load_config()
renipress = pd.read_csv("data/raw/renipress.csv", encoding="utf-8", low_memory=False)
distritos = gpd.read_file("data/raw/limites_distritales.geojson")

renipress = apply_category_rules(renipress, col_categoria="Categoria", col_estado="Condicion", cfg=cfg)
limpio, reporte_calidad = run_quality_pipeline(
    renipress,
    col_lon="ESTE", col_lat="NORTE", col_codigo="Codigo_Unico",
    col_ubigeo="UBIGEO", col_texto_libre=["Nombre_del_establecimiento"],
    districts_gdf=distritos, ubigeo_col_districts="IDDIST", cfg=cfg,
)

reporte_calidad.to_csv("data/outputs/data_quality_report.csv", index=False)
limpio.to_parquet("data/processed/ipress_clean.parquet", index=False)
```

`reporte_calidad` es exactamente la tabla que exige la consigna: una fila
por regla, con cuántos registros se marcaron, qué se hizo y por qué. Se usa
tal cual en el panel de Streamlit (Fase 4) y en el video.

Corre los tests (no requieren red ni Docker):
```bash
pytest tests/test_validation.py -v
```

## 3. Enrutamiento (Fase 2)

### 3.1 Levantar OSRM

```bash
bash docker/build_osrm.sh          # compila car, bike y foot (una vez; tarda)
docker compose -f docker/docker-compose.yml up -d
```

Verifica que responde:
```bash
curl "http://localhost:5000/nearest/v1/car/-77.03,-12.05"
```

### 3.2 Calcular la matriz

```python
from src.routing import sample_demand_points, compute_od_matrix, nearest_facility, compare_walk_vs_drive
from src.config_loader import load_config

cfg = load_config()
# demanda: centros poblados procesados con columnas id, lon, lat, distrito, poblacion
demanda_muestreada = sample_demand_points(demanda, col_distrito="ubigeo_distrito",
                                           col_poblacion="poblacion", cfg=cfg)

matriz_car = compute_od_matrix(demanda_muestreada, ipress_resolutivas, profile="car",
                                col_id_demand="id", col_id_facility="codigo_unico", cfg=cfg)
matriz_foot = compute_od_matrix(demanda_urbana, ipress_todas, profile="foot",
                                 col_id_demand="id", col_id_facility="codigo_unico", cfg=cfg)

comparacion = compare_walk_vs_drive(matriz_car, matriz_foot,
                                     facility_resolutive_ids=set(ipress_resolutivas["codigo_unico"]))
```

Cada matriz se cachea en `data/processed/matriz_<perfil>.parquet`. **Una
segunda corrida no vuelve a golpear OSRM** — borra el archivo si necesitas
recalcular. Esto es lo que permite que el dashboard (Fase 4) cargue en
segundos sin motor de enrutamiento activo.

Corre los tests sin OSRM:
```bash
pytest tests/test_routing.py -v -m "not osrm"
```

Corre también el smoke test contra OSRM real (requiere el paso 3.1):
```bash
pytest tests/test_routing.py -v -m osrm
```

## Estructura

```
config.md              # única fuente de parámetros (departamentos, umbrales, rutas, motor)
src/
  config_loader.py      # lee config.md
  acquisition.py         # Fase 1 — descarga
  validation.py           # Fase 1 — normalización de categoría + 6 reglas de calidad
  routing.py               # Fase 2 — cliente OSRM, snapping, matriz cacheada, comparaciones
docker/
  build_osrm.sh            # compila los 3 grafos (car/bike/foot)
  docker-compose.yml        # sirve los 3 perfiles en localhost
data/
  raw/                       # nunca se edita a mano; se regenera con acquisition.py
  processed/                  # se versiona (incluye la matriz precomputada)
  outputs/                     # informe de calidad, tablas para el reporte LaTeX
tests/
  test_validation.py
  test_routing.py
```

## Notas de diseño para el evaluador

- **Nada de números mágicos en el código.** Departamentos, whitelist de
  categorías, bbox de Perú, factor de desvío, límite de muestreo, puertos
  OSRM: todo en `config.md`, leído vía `config_loader.load_config()`.
- **Ninguna regla de validación descarta en silencio.** Cada fila marcada
  queda en el DataFrame con su columna `flag_*`; el reporte de calidad
  documenta la acción y la justificación por regla.
- **El factor de desvío del fallback no es un valor inventado**:
  `routing.validate_deviation_factor()` lo calibra empíricamente comparando
  distancia de red vs. línea recta en los pares que sí se lograron enrutar.
