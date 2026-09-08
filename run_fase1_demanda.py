import geopandas as gpd
from src.config_loader import load_config
from src.validation import run_quality_pipeline

cfg = load_config()

departamentos_permitidos = [
    cfg["departamentos"]["costero"]["nombre"],
    cfg["departamentos"]["andino"]["nombre"],
    cfg["departamentos"]["amazonico"]["nombre"],
]

# 1. Cargar centros poblados y filtrar a los 3 departamentos
cp = gpd.read_file("data/raw/CP_MED/CP_P.shp")
cp["DEP"] = cp["DEP"].str.strip().str.upper()
cp_3dep = cp[cp["DEP"].isin(departamentos_permitidos)].copy()
cp_3dep["UBIGEO"] = cp_3dep["UBIGEO"].astype(str).str.zfill(6)
cp_3dep["CODCP"] = cp_3dep["CODCP"].astype(str)
print(f"Centros poblados en los 3 departamentos: {len(cp_3dep)}")

# 2. Cargar límites distritales (mismos que usamos para RENIPRESS)
distritos = gpd.read_file("data/raw/limites_distritales.geojson")
distritos["NOMBDEP"] = distritos["NOMBDEP"].str.strip().str.upper()
distritos["IDDIST"] = distritos["IDDIST"].astype(str).str.zfill(6)
distritos_3dep = distritos[distritos["NOMBDEP"].isin(departamentos_permitidos)].copy()

# 3. Correr las 6 reglas completas de validación (las mismas que a RENIPRESS)
limpio, reporte_calidad = run_quality_pipeline(
    cp_3dep,
    col_lon="XGD", col_lat="YGD", col_codigo="CODCP",
    col_ubigeo="UBIGEO", col_texto_libre=["NOMCP", "MNOMCP"],
    districts_gdf=distritos_3dep, ubigeo_col_districts="IDDIST",
    cfg=cfg,
)

print("\n--- REPORTE DE CALIDAD (demanda) ---")
print(reporte_calidad.to_string(index=False))
print(f"\nFilas antes: {len(cp_3dep)}  |  Filas después: {len(limpio)}")

# 4. Renombrar a columnas estándar para routing.py y guardar como GeoParquet real
limpio_final = limpio.rename(
    columns={"CODCP": "id", "XGD": "lon", "YGD": "lat", "UBIGEO": "distrito"}
)
limpio_gdf = gpd.GeoDataFrame(
    limpio_final, geometry=gpd.points_from_xy(limpio_final["lon"], limpio_final["lat"]), crs="EPSG:4326"
)
limpio_gdf.to_parquet("data/processed/demanda_clean.parquet", index=False)

reporte_calidad.to_csv("data/outputs/data_quality_report_demanda.csv", index=False)
print("\nGuardado: data/processed/demanda_clean.parquet")
print("Guardado: data/outputs/data_quality_report_demanda.csv")