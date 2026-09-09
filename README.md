# Golden Hour — Acceso a salud resolutiva en Perú

Proyecto integrador: acceso real por carretera a centros de salud con
capacidad resolutiva (categoría II-1 en adelante) en tres departamentos:
**Piura** (costa), **Ayacucho** (andino), **Loreto** (amazónico).

Este README cubre las cinco fases. Ver `config.md` para cambiar
departamentos, umbrales o motor de enrutamiento sin tocar código.

**Ruta corta** (el repo ya trae las matrices y los outputs versionados, así
que las fases 4 y 5 corren sin OSRM ni Docker):

```bash
pip install -r requirements.txt
python run_fase4.py                 # prepara los artefactos del panel
streamlit run app.py      # panel interactivo (Fase 4)
python run_fase5.py                 # figuras y tablas del informe (Fase 5)
cd report && pdflatex informe.tex && pdflatex informe.tex
```

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

## 5. Panel interactivo (Fase 4)

```bash
python run_fase4.py                 # una vez: prepara los artefactos
streamlit run app.py
```

`run_fase4.py` no calcula ninguna métrica nueva: reutiliza
`metrics.attach_sample_weights`, `metrics.classify_urban_rural` y
`metrics.access_time` para dejar cinco artefactos que el panel abre en
segundos, **sin OSRM, sin Docker y sin geopandas**:

| Artefacto | Contenido |
|---|---|
| `data/processed/puntos_dashboard.parquet` | 1 fila por centro poblado (19,460), con población, zona, peso muestral, `t_min` y banda |
| `data/processed/instalaciones_dashboard.parquet` | 1 fila por IPRESS con coordenada válida (2,844), con categoría, institución y `es_resolutivo` |
| `data/processed/distritos_3dep.geojson` | los 225 polígonos distritales de los 3 departamentos (178 KB en vez de 1.9 MB) |
| `data/processed/mejoras_candidatos.parquet` | insumo del simulador: pares (centro poblado, candidato I-3/I-4) en los que ascender ese establecimiento mejoraría el tiempo actual |
| `data/outputs/calibracion_linea_recta.csv` | distancia de red vs. línea recta por departamento (+ muestra de pares para la figura) |
| `data/outputs/resumen_fase4.json` | trazabilidad de la corrida, que el panel muestra en la pestaña de metodología |

### Qué tiene el panel

- **Encabezado de indicadores** que se recalcula con cada filtro: población
  cubierta bajo el umbral elegido, población a más de 60 minutos, mediana
  ponderada del tiempo, peor distrito y Gini.
- **Mapa** con coropleto distrital por banda de acceso y capa de
  establecimientos (resolutivos y no resolutivos, en trazas separadas).
- **Distribución**: ECDF e histograma del tiempo, desagregables por
  departamento o por zona, con el umbral marcado.
- **Brechas**: tabla ordenable por cualquier columna y descargable a CSV, más
  el ranking de distritos críticos.
- **Simulador de escenarios**: se seleccionan uno o más establecimientos
  I-3/I-4 y el panel recalcula la cobertura con
  `metrics.apply_upgrades`, mostrando la ganancia marginal de población. Trae
  un ranking de candidatos por ganancia individual para no elegir a ciegas.
- **Equidad** (Lorenz + urbano/rural), **modos de viaje**, **altitud**,
  **calidad de datos** (los reportes de Fase 1 y 2 tal cual) y
  **metodología** con las descargas.

Filtros de la barra lateral: departamento, provincia, zona, umbral de tiempo,
categoría de establecimiento e institución (MINSA, EsSalud, Gobierno Regional,
privados, sanidades…).

**La decisión de diseño que importa:** el panel **no reimplementa ninguna
métrica**. Al mover un filtro vuelve a llamar a las funciones de
`src/metrics.py` sobre el subconjunto filtrado (`src/dashboard_data.py` se
encarga de devolverles los DataFrames en la forma que esperan). Con todo
seleccionado, los números del panel son idénticos a los CSV de
`data/outputs/` que cita el informe — y hay un test que lo comprueba:

```bash
pytest tests/test_dashboard.py -v                       # todo
pytest tests/test_dashboard.py -v -m "not artefactos"    # sin correr run_fase4.py
```

### Nota sobre los tiempos del simulador

La matriz OD de Fase 2 se calculó contra los 58 establecimientos
**resolutivos**, no contra los 607 candidatos a ascenso. Mientras no exista
`data/processed/matriz_car_candidatos.parquet`, el simulador estima esos
tiempos con la calibración empírica línea-recta/red de cada departamento, los
marca como `estimado` y lo advierte en pantalla. Para calcularlos de verdad,
con OSRM levantado:

```python
from src.config_loader import load_config, path_for
from src.routing import compute_od_matrix, sample_demand_points
import pandas as pd

cfg = load_config()
demanda = pd.read_parquet(path_for("demanda_processed", cfg))
ipress = pd.read_parquet(path_for("ipress_processed", cfg)).rename(
    columns={"ESTE": "lon", "NORTE": "lat", "COD_IPRESS": "id"})
candidatos = ipress[ipress["categoria_norm"].isin(["I-3", "I-4"])]
muestra = sample_demand_points(demanda, col_distrito="distrito", col_poblacion=None, cfg=cfg)

compute_od_matrix(muestra, candidatos, profile="car",
                  col_id_demand="id", col_id_facility="id",
                  cache_path=path_for("processed", cfg) / "matriz_car_candidatos.parquet",
                  cfg=cfg)
```

`run_fase4.py` detecta ese archivo y lo usa en lugar de la estimación.

## 6. Informe y figuras (Fase 5)

```bash
python run_fase5.py
cd report && pdflatex informe.tex && pdflatex informe.tex
```

`run_fase5.py` lee los CSV de `data/outputs/` (misma fuente de verdad que el
panel) y genera:

- `report/figures/*.pdf` (y `.png`) — 8 figuras: mapa coroplético por
  departamento, cobertura por banda, línea recta vs. red, curva de Lorenz,
  contraste urbano/rural, ranking de distritos críticos, comparación de modos
  y dispersión altitud/tiempo. El PDF es vectorial (es el que cita el
  informe); el PNG a 200 dpi es para el README y el video.
- `report/tables/informe_*.tex` — 9 tablas `booktabs` con encabezados en
  castellano y **contenido escapado**. Las tablas de Fase 3
  (`report/tables/*.tex` sin prefijo) se conservan intactas como volcado
  reproducible, pero no compilan dentro de un documento: sus encabezados son
  nombres de variable con guiones bajos, que LaTeX interpreta.
- `report/informe.tex` — el informe (12 páginas), que incluye una sección de
  **Limitaciones** con las doce que pueden alterar las conclusiones,
  ordenadas por impacto y con la dirección del sesgo cuando se conoce.
- `report/guion_video.md` — guion de la presentación en video, minuto a
  minuto, apoyado en el panel.

El PDF compilado (`report/informe.pdf`) está versionado para que no haga
falta una distribución LaTeX solo para leerlo.

## Estructura

```
config.md              # única fuente de parámetros (departamentos, umbrales, rutas, motor)
src/
  config_loader.py      # lee config.md
  acquisition.py         # Fase 1 — descarga
  validation.py           # Fase 1 — normalización de categoría + 6 reglas de calidad
  population.py            # Fase 3 — población censada 2017 por centro poblado
  routing.py                # Fase 2 — cliente OSRM, snapping, matriz cacheada, comparaciones
  metrics.py                 # Fase 3 — métricas puras (sin I/O)
  export.py                   # Fase 3 — escribe CSV + tablas LaTeX
  dashboard_data.py            # Fase 4 — carga artefactos y los adapta al contrato de metrics.py
app.py                          # Fase 4 — panel Streamlit (solo presentación)
run_fase1.py … run_fase5.py       # un script por fase, ejecutables en orden
docker/
  build_osrm.sh                   # compila los 3 grafos (car/bike/foot)
  docker-compose.yml               # sirve los 3 perfiles en localhost
data/
  raw/                              # nunca se edita a mano; se regenera con acquisition.py
  processed/                         # se versiona (incluye la matriz precomputada)
  outputs/                            # reportes de calidad y tablas de métricas
report/
  informe.tex / informe.pdf            # Fase 5 — informe final
  figures/ · tables/                    # generados por run_fase5.py
  guion_video.md                         # guion de la presentación
tests/
  test_validation.py · test_routing.py · test_metrics.py
  test_dashboard.py · test_informe.py
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
- **El panel no puede contradecir al informe.** No hay una segunda
  implementación de "media ponderada" o "banda de cobertura" en la capa de
  presentación: `app.py` llama a `src/metrics.py`, igual que
  `run_fase3.py`. `tests/test_dashboard.py` verifica que recalcular desde los
  artefactos del panel reproduce exactamente `coverage_bands.csv`.
- **El simulador es la razón de haber guardado la matriz completa.** Con solo
  la instalación más cercana por punto no se puede responder "¿y si este I-3
  fuera resolutivo?": haría falta volver a enrutar. `metrics.apply_upgrades`
  recalcula el mínimo sobre la matriz ya almacenada.
- **El factor de desvío del respaldo se recalibró y estaba mal.** El empírico
  es 1.59 (1.41 en Ayacucho, 1.87 en Loreto), no el 1.35 que traía
  `config.md`; y la diferencia grande entre departamentos no es la forma de
  la red sino su velocidad: 63 km/h en la sierra contra 12.7 en la selva.
- **Cada cifra dice de qué universo habla.** Las métricas describen la muestra
  enrutada de 5,002 centros poblados (de 19,460), ponderada por población
  censada y por el peso de diseño del muestreo. El panel lo repite en cada
  vista y el informe lo cuantifica en Limitaciones (incluida la sobreestimación
  de ~17 % de la población total por expansión de pesos).
- **Un gris no es un dato malo.** En los mapas, "sin ruta en la red vial"
  (un punto sin acceso enrutable) y "sin población censada" (una unidad sin
  peso para promediar) son categorías distintas y con colores distintos: ver
  `dashboard_data.assign_bands_medias()`.
