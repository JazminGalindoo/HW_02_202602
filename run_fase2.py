import pandas as pd
from src.config_loader import load_config, path_for
from src.routing import (
    sample_demand_points, snap_points, compute_od_matrix,
    compare_walk_vs_drive, compare_all_modes, validate_deviation_factor,
)

cfg = load_config()

# 1. Cargar demanda e instalaciones ya validadas en Fase 1
demanda = pd.read_parquet("data/processed/demanda_clean.parquet")
ipress = pd.read_parquet("data/processed/ipress_clean.parquet")
ipress = ipress.rename(columns={"ESTE": "lon", "NORTE": "lat", "COD_IPRESS": "id"})
resolutivas = ipress[ipress["es_resolutivo"]].copy()

print(f"Instalaciones resolutivas: {len(resolutivas)}")
print(f"Instalaciones totales (para comparación a pie): {len(ipress)}")
print(f"Centros poblados totales: {len(demanda)}")

# 2. Muestreo a máximo 5,000 puntos (estratificado por distrito)
demanda_muestra = sample_demand_points(
    demanda, col_distrito="distrito", col_poblacion=None, cfg=cfg
)
print(f"Puntos de demanda tras muestreo: {len(demanda_muestra)}")

# 3. Snapping de la demanda muestreada al perfil "car" (reporte requerido)
resumenes_snap = []
_, r = snap_points(demanda_muestra, col_lon="lon", col_lat="lat", col_id="id", profile="car", cfg=cfg)
resumenes_snap.append({**r, "dataset": "demanda_muestreada"})
_, r = snap_points(demanda_muestra, col_lon="lon", col_lat="lat", col_id="id", profile="bike", cfg=cfg)
resumenes_snap.append({**r, "dataset": "demanda_muestreada"})
_, r = snap_points(demanda_muestra, col_lon="lon", col_lat="lat", col_id="id", profile="foot", cfg=cfg)
resumenes_snap.append({**r, "dataset": "demanda_muestreada"})
_, r = snap_points(resolutivas, col_lon="lon", col_lat="lat", col_id="id", profile="car", cfg=cfg)
resumenes_snap.append({**r, "dataset": "instalaciones_resolutivas"})
pd.DataFrame(resumenes_snap).to_csv("data/outputs/snapping_report.csv", index=False)
print("\n--- Reporte de snapping (auto/bici/pie en demanda, auto en instalaciones) ---")
print(pd.DataFrame(resumenes_snap).to_string(index=False))

# 4. Matrices: demanda muestreada x instalaciones RESOLUTIVAS, en los 3 perfiles
print("\n=== Matriz CAR (demanda x resolutivas) ===")
matriz_car = compute_od_matrix(
    demanda_muestra, resolutivas, profile="car",
    col_id_demand="id", col_id_facility="id", cfg=cfg,
)
print("\n=== Matriz BIKE (demanda x resolutivas) ===")
matriz_bike = compute_od_matrix(
    demanda_muestra, resolutivas, profile="bike",
    col_id_demand="id", col_id_facility="id", cfg=cfg,
)
print("\n=== Matriz FOOT (demanda x resolutivas) ===")
matriz_foot_resolutivas = compute_od_matrix(
    demanda_muestra, resolutivas, profile="foot",
    col_id_demand="id", col_id_facility="id", cfg=cfg,
)

resolutivas_ids = set(resolutivas["id"])

comp_modos = compare_all_modes(
    {"car": matriz_car, "bike": matriz_bike, "foot": matriz_foot_resolutivas},
    resolutivas_ids,
)
comp_modos.to_csv("data/outputs/comparacion_modos.csv", index=False)
print("\n--- Comparación de los 3 modos (todos los muestreados) ---")
print(comp_modos.describe())

# 5. "Urbano" = cualquier nivel de capital (departamento, provincia o distrito)
#    Usamos TODOS los puntos urbanos reales (no el muestreo), porque son pocos
#    (237) y perderlos por azar en el muestreo estratificado desvirtuaría la
#    comparación pie vs coche que pide la consigna.
demanda["CAPITAL"] = pd.to_numeric(demanda["CAPITAL"], errors="coerce").fillna(0)
urbanos = demanda[demanda["CAPITAL"].isin([1, 2, 3])].copy()
print(f"\nPuntos urbanos (capital dep/prov/distrito): {len(urbanos)}")

print("\n=== CAR: urbanos x resolutivas ===")
matriz_car_urbanos = compute_od_matrix(
    urbanos, resolutivas, profile="car",
    col_id_demand="id", col_id_facility="id",
    cache_path=path_for("processed", cfg) / "matriz_car_urbanos.parquet",
    cfg=cfg,
)
print("\n=== FOOT: urbanos x TODAS las instalaciones ===")
matriz_foot_urbanos_todas = compute_od_matrix(
    urbanos, ipress, profile="foot",
    col_id_demand="id", col_id_facility="id",
    cache_path=path_for("matriz_foot_urbanos_todas", cfg),
    cfg=cfg,
)

comp_walk_drive = compare_walk_vs_drive(matriz_car_urbanos, matriz_foot_urbanos_todas, resolutivas_ids)
comp_walk_drive.to_csv("data/outputs/comparacion_pie_vs_coche.csv", index=False)

print("\n--- Comparación pie vs coche (puntos urbanos reales) ---")
print(f"Total puntos comparados: {len(comp_walk_drive)}")
print(f"Cambia la instalación más cercana según el modo: "
      f"{(~comp_walk_drive['misma_instalacion'].fillna(False)).sum()} de {len(comp_walk_drive)}")
print(f"Inalcanzables a pie: {comp_walk_drive['min_foot'].isna().sum()}")
print("\nRatio pie/coche (percentiles):")
print(comp_walk_drive["ratio_foot_car"].describe())

# 6. Factor de desvío empírico (para el fallback documentado)
factor = validate_deviation_factor(matriz_car, demanda_muestra, resolutivas, "id", "id")
print("\n--- Factor de desvío empírico (red / línea recta), perfil car ---")
print(factor)

print("\nListo. Todo guardado en data/processed/ y data/outputs/")
# 7. Aplicar el fallback documentado a los puntos que NO se pudieron enrutar
#    en auto -- en vez de dejarlos simplemente excluidos.
from src.routing import nearest_facility, apply_fallback

nearest_car = nearest_facility(matriz_car, resolutivas_ids)
ids_con_ruta = set(nearest_car["id_demanda"])
ids_todos = set(demanda_muestra["id"])
ids_sin_ruta = list(ids_todos - ids_con_ruta)
print(f"\nPuntos sin ruta en auto: {len(ids_sin_ruta)} de {len(ids_todos)}")

nearest_car_completo = apply_fallback(
    nearest_car, demanda_muestra, resolutivas,
    unroutable_demand_ids=ids_sin_ruta,
    col_id_demand="id", col_id_facility="id",
    factor_desvio=factor["factor_mediano"],
)
nearest_car_completo.to_csv("data/outputs/nearest_car_con_fallback.csv", index=False)
print(f"Guardado: data/outputs/nearest_car_con_fallback.csv "
      f"({nearest_car_completo['estimado_por_fallback'].sum()} filas estimadas por fallback, "
      f"marcadas explícitamente)")