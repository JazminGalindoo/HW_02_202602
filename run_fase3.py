import pandas as pd

from src.config_loader import load_config, path_for
from src.population import build_poblacion_centros_poblados, merge_population_to_demand
from src.metrics import (
    classify_urban_rural, attach_sample_weights, access_time, coverage_bands,
    weighted_mean_access, critical_gap_ranking, gini_access, urban_rural_contrast,
    access_vs_altitude,
)
from src.export import export_csv, export_latex

cfg = load_config()

# 1. Cargar demanda con población (Tarea 1). Si no existe aún, se genera.
demanda_con_poblacion_path = path_for("demanda_con_poblacion", cfg)
if demanda_con_poblacion_path.exists():
    demanda_con_poblacion = pd.read_parquet(demanda_con_poblacion_path)
else:
    demanda = pd.read_parquet(path_for("demanda_processed", cfg))
    poblacion = build_poblacion_centros_poblados(cfg)
    demanda_con_poblacion, _ = merge_population_to_demand(demanda, poblacion, cfg=cfg)
    demanda_con_poblacion.to_parquet(demanda_con_poblacion_path, index=False)

ipress = pd.read_parquet(path_for("ipress_processed", cfg))
resolutivas_ids = set(ipress.loc[ipress["es_resolutivo"], "COD_IPRESS"])
matriz_car = pd.read_parquet(path_for("matriz_car", cfg))

print(f"Puntos de demanda (universo): {len(demanda_con_poblacion)}")
print(f"Instalaciones resolutivas: {len(resolutivas_ids)}")
print(f"Filas en matriz_car: {len(matriz_car)}")

# 2. Tarea 2: clasificación urbano/rural + pesos de muestreo de Fase 2,
#    combinados en un solo population_df para todas las métricas.
demanda_pesos = attach_sample_weights(demanda_con_poblacion, cfg)
population_df = classify_urban_rural(demanda_pesos)

# 3. Tarea 3: métricas
access_df = access_time(matriz_car, resolutivas_ids)

cb = coverage_bands(access_df, population_df, cfg)
print("\n--- Bandas de acceso (% población, muestra enrutada) ---")
print(cb.to_string(index=False))

wma_distrito = weighted_mean_access(access_df, population_df, level="distrito", cfg=cfg)
wma_provincia = weighted_mean_access(access_df, population_df, level="provincia", cfg=cfg)
wma_departamento = weighted_mean_access(access_df, population_df, level="departamento", cfg=cfg)
print("\n--- t_min medio ponderado por departamento ---")
print(wma_departamento.to_string(index=False))

ranking = critical_gap_ranking(wma_distrito, cfg=cfg)
print("\n--- Top distritos con peor acceso ---")
print(ranking[["ranking", "nivel_nombre", "t_min_medio_ponderado", "poblacion_con_acceso"]].to_string(index=False))

gini_resumen, gini_lorenz = gini_access(access_df, population_df)
print("\n--- Gini de acceso ---")
print(gini_resumen.to_string(index=False))

urc = urban_rural_contrast(access_df, population_df)
print("\n--- Contraste urbano vs. rural ---")
print(urc.to_string(index=False))

alt_por_punto, alt_resumen = access_vs_altitude(access_df, demanda_con_poblacion)
print("\n--- Acceso vs. altitud ---")
print(alt_resumen.drop(columns="nota_causalidad").to_string(index=False))
print(f"NOTA: {alt_resumen['nota_causalidad'].iloc[0]}")

# 4. Tarea 4: exportar. Todo a CSV; el subconjunto citable en el informe
#    también a LaTeX (booktabs).
export_csv(cb, "coverage_bands", cfg)
export_csv(wma_distrito, "weighted_mean_access_distrito", cfg)
export_csv(wma_provincia, "weighted_mean_access_provincia", cfg)
export_csv(wma_departamento, "weighted_mean_access_departamento", cfg)
export_csv(ranking, "critical_gap_ranking", cfg)
export_csv(gini_resumen, "gini_access_resumen", cfg)
export_csv(gini_lorenz, "gini_access_lorenz_curve", cfg)
export_csv(urc, "urban_rural_contrast", cfg)
export_csv(alt_por_punto, "access_vs_altitude_por_punto", cfg)
export_csv(alt_resumen, "access_vs_altitude_resumen", cfg)

export_latex(cb, "coverage_bands", cfg)
export_latex(wma_departamento, "weighted_mean_access_departamento", cfg)
export_latex(ranking.drop(columns="poblacion_minima_aplicada"), "critical_gap_ranking", cfg)
export_latex(gini_resumen, "gini_access_resumen", cfg)
export_latex(urc, "urban_rural_contrast", cfg)
# La nota de causalidad es texto largo -> no entra a una tabla LaTeX; se cita en prosa en el informe.
export_latex(alt_resumen.drop(columns="nota_causalidad"), "access_vs_altitude_resumen", cfg)

print("\nListo. CSV en data/outputs/, tablas LaTeX en report/tables/.")
