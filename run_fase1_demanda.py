import geopandas as gpd
from src.config_loader import load_config
from src.validation import flag_missing_coords, flag_out_of_bbox

cfg = load_config()

departamentos_permitidos = [
    cfg["departamentos"]["costero"]["nombre"],
    cfg["departamentos"]["andino"]["nombre"],
    cfg["departamentos"]["amazonico"]["nombre"],
]

cp = gpd.read_file("data/raw/CP_MED/CP_P.shp")
print(f"Centros poblados nacional: {len(cp)}")

cp["DEP"] = cp["DEP"].str.strip().str.upper()
cp_3dep = cp[cp["DEP"].isin(departamentos_permitidos)].copy()
print(f"Centros poblados en los 3 departamentos: {len(cp_3dep)}")
print(cp_3dep["DEP"].value_counts())

# Validación básica de coordenadas (reutilizamos las reglas de Fase 1)
mask_missing = flag_missing_coords(cp_3dep, "XGD", "YGD")
mask_bbox = flag_out_of_bbox(cp_3dep, "XGD", "YGD", cfg)
print(f"\nCoords faltantes: {mask_missing.sum()}")
print(f"Fuera de bbox Perú: {mask_bbox.sum()}")

cp_3dep_clean = cp_3dep[~mask_missing & ~mask_bbox].copy()
cp_3dep_clean["UBIGEO"] = cp_3dep_clean["UBIGEO"].astype(str).str.zfill(6)
print(f"\nCentros poblados limpios: {len(cp_3dep_clean)}")

# Renombramos a columnas estándar que routing.py (Fase 2) espera
cp_3dep_clean = cp_3dep_clean.rename(
    columns={"CODCP": "id", "XGD": "lon", "YGD": "lat", "UBIGEO": "distrito"}
)

cp_3dep_clean.to_parquet("data/processed/demanda_clean.parquet", index=False)
print("\nGuardado: data/processed/demanda_clean.parquet")