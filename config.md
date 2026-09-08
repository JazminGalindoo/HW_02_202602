# Configuración del proyecto — Golden Hour

Este archivo es la **única fuente de verdad** para rutas, departamentos,
listas blancas de categoría, umbrales y selección de motor de enrutamiento.
Ningún módulo en `src/` debe tener estos valores escritos directamente en el código
("hardcoded"); todos se leen desde el bloque YAML de abajo mediante
`src/config_loader.py`.

Para cambiar de departamento, umbral o motor de enrutamiento: edita el YAML
de aquí abajo. No hace falta tocar ningún `.py`.

```yaml
# ---------------------------------------------------------------------------
# ALCANCE GEOGRÁFICO
# ---------------------------------------------------------------------------
departamentos:
  costero:
    nombre: "PIURA"
    ubigeo_dep: "20"
  andino:
    nombre: "AYACUCHO"
    ubigeo_dep: "05"
  amazonico:
    nombre: "LORETO"
    ubigeo_dep: "16"

# Caja delimitadora aproximada del Perú (para la validación de coordenadas)
bbox_peru:
  lon_min: -81.4
  lon_max: -68.6
  lat_min: -18.4
  lat_max: -0.04

# ---------------------------------------------------------------------------
# DEFINICIÓN DE CAPACIDAD RESOLUTIVA (RENIPRESS)
# ---------------------------------------------------------------------------
categorias_resolutivas:
  - "II-1"
  - "II-2"
  - "II-E"
  - "III-1"
  - "III-2"
  - "III-E"

categorias_no_resolutivas:
  - "I-1"
  - "I-2"
  - "I-3"
  - "I-4"

# Valores crudos observados en RENIPRESS que deben normalizarse a cada
# categoría canónica. Las cadenas reales en el registro traen variaciones de
# mayúsculas, espacios, guiones, y a veces el sufijo "CATEGORIA" repetido.
# Esta tabla es la ÚNICA fuente de reglas de normalización de categoría;
# si el registro trae algo que no matchea ni por regex, se marca como
# 'SIN_CATEGORIA' y se conserva con advertencia (no se descarta).
normalizacion_categoria:
  "I-1": ["I-1", "I1", "CATEGORIA I-1", "I - 1"]
  "I-2": ["I-2", "I2", "CATEGORIA I-2", "I - 2"]
  "I-3": ["I-3", "I3", "CATEGORIA I-3", "I - 3"]
  "I-4": ["I-4", "I4", "CATEGORIA I-4", "I - 4"]
  "II-1": ["II-1", "II1", "CATEGORIA II-1", "II - 1"]
  "II-2": ["II-2", "II2", "CATEGORIA II-2", "II - 2"]
  "II-E": ["II-E", "IIE", "CATEGORIA II-E", "II - E"]
  "III-1": ["III-1", "III1", "CATEGORIA III-1", "III - 1"]
  "III-2": ["III-2", "III2", "CATEGORIA III-2", "III - 2"]
  "III-E": ["III-E", "IIIE", "CATEGORIA III-E", "III - E"]

# Valores crudos de la columna de estado/condición que cuentan como "activo".
# RENIPRESS usa históricamente "EN FUNCIONAMIENTO" pero hay variantes.
estados_activos:
  - "EN FUNCIONAMIENTO"
  - "ACTIVO"
  - "ACTIVADO"

# ---------------------------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------------------------
validacion:
  tolerancia_fuera_de_poligono_m: 500   # margen antes de marcar "fuera de su distrito"
  encoding_candidatos: ["utf-8", "latin-1", "cp1252"]

# ---------------------------------------------------------------------------
# RUTAS (todo relativo a la raíz del repo)
# ---------------------------------------------------------------------------
rutas:
  raw: "data/raw"
  processed: "data/processed"
  outputs: "data/outputs"
  logs: "logs"

  renipress_raw: "data/raw/renipress.csv"
  centros_poblados_raw: "data/raw/centros_poblados.gpkg"
  osm_pbf_raw: "data/raw/peru-latest.osm.pbf"
  limites_distritales_raw: "data/raw/limites_distritales.geojson"

  ipress_processed: "data/processed/ipress_clean.parquet"
  demanda_processed: "data/processed/demanda_clean.parquet"
  matriz_car: "data/processed/matriz_car.parquet"
  matriz_foot: "data/processed/matriz_foot.parquet"
  matriz_bike: "data/processed/matriz_bike.parquet"
  snapping_report: "data/outputs/snapping_report.csv"
  quality_report: "data/outputs/data_quality_report.csv"

# ---------------------------------------------------------------------------
# FUENTES (URLs / identificadores de dataset)
# ---------------------------------------------------------------------------
fuentes:
  renipress:
    portal: "https://www.datosabiertos.gob.pe"
    dataset_slug: "registro-nacional-de-entidades-prestadoras-de-servicios-de-salud-renipress"
    # El portal es DKAN (compatible con la API de CKAN). Se resuelve la URL
    # real del recurso CSV en tiempo de ejecución vía package_show, en vez de
    # apuntar a un enlace de descarga que puede cambiar de UUID.
    api_package_show: "https://www.datosabiertos.gob.pe/api/3/action/package_show?id={slug}"
    fecha_descarga_cache: null   # se completa automáticamente al descargar

  sigmed_centros_poblados:
    portal: "https://sigmed.minedu.gob.pe/descargas/"
    # SIGMED no expone un endpoint estable de API pública; requiere navegar
    # el portal y confirmar el enlace del recurso vigente. Ver README para
    # el procedimiento manual de un solo paso (una vez, no en cada corrida).
    url_manual: null
    fecha_descarga_cache: null

  osm_peru:
    url: "https://download.geofabrik.de/south-america/peru-latest.osm.pbf"
    fecha_descarga_cache: null

  limites_administrativos:
    # Repo público con polígonos de departamento/provincia/distrito derivados
    # de INEI 2007 (georreferenciados). Fuente citada en el informe.
    repo: "https://github.com/juaneladio/peru-geojson"
    url_distritos: "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"
    fecha_descarga_cache: null

# ---------------------------------------------------------------------------
# ENRUTAMIENTO (Fase 2)
# ---------------------------------------------------------------------------
enrutamiento:
  motor: "osrm"     # "osrm" | "osmnx" — cambiar aquí basta para conmutar de motor
  perfiles: ["car", "bike", "foot"]
  osrm:
    host: "localhost"
    puertos:
      car: 5000
      bike: 5001
      foot: 5002
    timeout_s: 10
    reintentos: 3
    espera_entre_reintentos_s: 2
  snapping:
    radio_max_m: 1000   # si un punto no encuentra red vial en este radio, se marca "no-snap"
  muestreo:
    limite_puntos_demanda: 5000
    estrategia: "estratificada_por_distrito"   # ponderada por población dentro de cada distrito
    semilla_aleatoria: 42
  fallback_no_enrutable:
    permitir_linea_recta: true
    factor_desvio_default: 1.35   # debe recalibrarse empíricamente (ver validate_deviation_factor en routing.py)
```
