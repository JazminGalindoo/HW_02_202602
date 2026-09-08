import pandas as pd
import geopandas as gpd
from src.config_loader import load_config
from src.validation import apply_category_rules, run_quality_pipeline

cfg = load_config()

departamentos_permitidos = [
    cfg["departamentos"]["costero"]["nombre"],
    cfg["departamentos"]["andino"]["nombre"],
    cfg["departamentos"]["amazonico"]["nombre"],
]

# 1. Cargar RENIPRESS, forzando UBIGEO como texto (si no, Python le come el
#    cero inicial a los ubigeos de Ayacucho, que empiezan con "05")
renipress = pd.read_csv(
    "data/raw/renipress.csv", sep=";", encoding="utf-8-sig", low_memory=False,
    dtype={"UBIGEO": str, "COD_IPRESS": str},
)
renipress["UBIGEO"] = renipress["UBIGEO"].str.zfill(6)
renipress["DEPARTAMENTO"] = renipress["DEPARTAMENTO"].str.strip().str.upper()

renipress_3dep = renipress[renipress["DEPARTAMENTO"].isin(departamentos_permitidos)].copy()
print(f"RENIPRESS en los 3 departamentos: {len(renipress_3dep)} filas")

# 2. Normalizar categoría y resolutividad
renipress_3dep = apply_category_rules(renipress_3dep, col_categoria="CATEGORIA", col_estado="ESTADO", cfg=cfg)

# 3. Cargar y filtrar límites distritales a los mismos 3 departamentos
distritos = gpd.read_file("data/raw/limites_distritales.geojson")
distritos["NOMBDEP"] = distritos["NOMBDEP"].str.strip().str.upper()
distritos["IDDIST"] = distritos["IDDIST"].astype(str).str.zfill(6)
distritos_3dep = distritos[distritos["NOMBDEP"].isin(departamentos_permitidos)].copy()
print(f"Distritos en los 3 departamentos: {len(distritos_3dep)}")

# 4. Correr las 6 reglas de validación
limpio, reporte_calidad = run_quality_pipeline(
    renipress_3dep,
    col_lon="ESTE", col_lat="NORTE", col_codigo="COD_IPRESS",
    col_ubigeo="UBIGEO", col_texto_libre=["NOMBRE", "DIRECCION"],
    districts_gdf=distritos_3dep, ubigeo_col_districts="IDDIST",
    cfg=cfg,
)

print("\n--- REPORTE DE CALIDAD ---")
print(reporte_calidad.to_string(index=False))

print(f"\nFilas antes de validar: {len(renipress_3dep)}")
print(f"Filas después de validar (data/processed): {len(limpio)}")
print(f"Resolutivas en el set limpio: {limpio['es_resolutivo'].sum()}")

# 5. Exportar
reporte_calidad.to_csv("data/outputs/data_quality_report.csv", index=False)
limpio.to_parquet("data/processed/ipress_clean.parquet", index=False)
print("\nGuardado: data/outputs/data_quality_report.csv")
print("Guardado: data/processed/ipress_clean.parquet")