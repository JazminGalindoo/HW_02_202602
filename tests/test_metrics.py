"""Tests de las métricas de Fase 3. No requieren red ni los parquet reales:
son puramente sobre DataFrames sintéticos pequeños con valores elegidos para
poder verificar los resultados a mano (mismo estilo que test_validation.py).

Escenario compartido (3 puntos de demanda, 2 instalaciones, 1 resolutiva):
    d1: distrito 010101 (DEP1/PROV_UNO), CAPITAL='1' (capital dep -> urbano),
        Z=100,  población=100. Llega a f1 (resolutiva) en 10 min.
    d2: distrito 010101 (DEP1/PROV_UNO), CAPITAL='0' (rural),
        Z=2000, población=300. Llega a f1 en 50 min.
    d3: distrito 010201 (DEP1/PROV_DOS), CAPITAL='0' (rural),
        Z=3000, población=50.  NO llega a f1 (no enrutable).
    f2 no es resolutiva -- cualquier ruta hacia f2 debe ser ignorada por
    access_time.
"""

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.metrics import (
    classify_urban_rural, attach_sample_weights, access_time, coverage_bands,
    weighted_mean_access, critical_gap_ranking, gini_access, urban_rural_contrast,
    access_vs_altitude, weighted_median_access, population_within, apply_upgrades,
    ranking_upgrade_gain,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def matrix_df():
    return pd.DataFrame({
        "id_demanda":     ["d1", "d1", "d2", "d2", "d3", "d3"],
        "id_instalacion": ["f1", "f2", "f1", "f2", "f1", "f2"],
        "perfil":         ["car"] * 6,
        "duracion_min":   [10.0, 5.0, 50.0, 1.0, np.nan, 1.0],
        "distancia_km":   [5.0, 2.0, 30.0, 1.0, np.nan, 1.0],
        "enrutable":      [True, True, True, True, False, True],
    })


@pytest.fixture
def facility_resolutive_ids():
    return {"f1"}


@pytest.fixture
def population_df():
    return pd.DataFrame({
        "id": ["d1", "d2", "d3"],
        "poblacion_censada": [100.0, 300.0, 50.0],
        "distrito": ["010101", "010101", "010201"],
        "PROV": ["PROV_UNO", "PROV_UNO", "PROV_DOS"],
        "DEP": ["DEP1", "DEP1", "DEP1"],
        "DIST": ["DIST_A", "DIST_A", "DIST_B"],
        "CAPITAL": ["1", "0", "0"],
    })


@pytest.fixture
def demanda_clean_like():
    return pd.DataFrame({
        "id": ["d1", "d2", "d3"],
        "Z": [100, 2000, 3000],
    })


# ---------------------------------------------------------------------------
# Tarea 2 — classify_urban_rural
# ---------------------------------------------------------------------------

def test_classify_urban_rural_regla_capital(population_df):
    out = classify_urban_rural(population_df)
    assert out.set_index("id")["zona"].to_dict() == {"d1": "urbano", "d2": "rural", "d3": "rural"}


def test_classify_urban_rural_capital_nulo_no_crashea_y_es_rural():
    df = pd.DataFrame({"id": ["x"], "CAPITAL": [None]})
    out = classify_urban_rural(df)
    assert out["zona"].iloc[0] == "rural"


def test_classify_urban_rural_requiere_columna_capital():
    with pytest.raises(KeyError):
        classify_urban_rural(pd.DataFrame({"id": ["x"]}))


# ---------------------------------------------------------------------------
# access_time
# ---------------------------------------------------------------------------

def test_access_time_toma_solo_instalaciones_resolutivas_y_flagea_sin_ruta(matrix_df, facility_resolutive_ids):
    out = access_time(matrix_df, facility_resolutive_ids).set_index("id_demanda")

    assert out.loc["d1", "t_min"] == 10.0
    assert out.loc["d1", "flag_sin_acceso_enrutable"] == False  # noqa: E712
    assert out.loc["d2", "t_min"] == 50.0
    assert pd.isna(out.loc["d3", "t_min"])
    assert out.loc["d3", "flag_sin_acceso_enrutable"] == True  # noqa: E712


def test_access_time_universo_es_el_de_la_matriz_no_facility_no_resolutiva(matrix_df, facility_resolutive_ids):
    out = access_time(matrix_df, facility_resolutive_ids)
    assert set(out["id_demanda"]) == {"d1", "d2", "d3"}
    # d1 llega a f2 en 5 min, pero f2 no es resolutiva -> debe ignorarse
    assert out.set_index("id_demanda").loc["d1", "t_min"] == 10.0


# ---------------------------------------------------------------------------
# coverage_bands
# ---------------------------------------------------------------------------

def test_coverage_bands_asigna_bandas_y_suma_100(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    reporte = coverage_bands(access_df, population_df, cfg).set_index("banda")

    # población total ponderada = 100+300+50 = 450
    assert reporte.loc["0-30", "poblacion"] == pytest.approx(100.0)   # d1, t=10
    assert reporte.loc["30-60", "poblacion"] == pytest.approx(300.0)  # d2, t=50
    assert reporte.loc["60-120", "poblacion"] == pytest.approx(0.0)
    assert reporte.loc[">120", "poblacion"] == pytest.approx(0.0)
    assert reporte.loc["sin_acceso_enrutable", "poblacion"] == pytest.approx(50.0)  # d3
    assert reporte["pct_poblacion"].sum() == pytest.approx(100.0)


def test_coverage_bands_sin_poblacion_conocida_no_crashea(matrix_df, facility_resolutive_ids, cfg):
    pop_vacia = pd.DataFrame({"id": ["d1", "d2", "d3"], "poblacion_censada": [np.nan, np.nan, np.nan]})
    access_df = access_time(matrix_df, facility_resolutive_ids)
    reporte = coverage_bands(access_df, pop_vacia, cfg)
    assert (reporte["pct_poblacion"] == 0.0).all()


# ---------------------------------------------------------------------------
# weighted_mean_access
# ---------------------------------------------------------------------------

def test_weighted_mean_access_nivel_distrito(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    out = weighted_mean_access(access_df, population_df, level="distrito", cfg=cfg).set_index("nivel_id")

    # 010101: (10*100 + 50*300) / 400 = 40.0, ambos con acceso
    assert out.loc["010101", "t_min_medio_ponderado"] == pytest.approx(40.0)
    assert out.loc["010101", "poblacion_con_acceso"] == pytest.approx(400.0)
    assert out.loc["010101", "pct_poblacion_con_acceso"] == pytest.approx(100.0)

    # 010201: solo d3, sin acceso enrutable -> media NaN, toda la población "sin acceso"
    assert pd.isna(out.loc["010201", "t_min_medio_ponderado"])
    assert out.loc["010201", "poblacion_sin_acceso_enrutable"] == pytest.approx(50.0)
    assert out.loc["010201", "pct_poblacion_con_acceso"] == pytest.approx(0.0)


def test_weighted_mean_access_nivel_departamento_agrega_todo(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    out = weighted_mean_access(access_df, population_df, level="departamento", cfg=cfg)

    assert len(out) == 1  # los 3 puntos están en el mismo departamento (DEP1, ubigeo '01')
    fila = out.iloc[0]
    assert fila["nivel_id"] == "01"
    assert fila["t_min_medio_ponderado"] == pytest.approx(40.0)
    assert fila["poblacion_sin_acceso_enrutable"] == pytest.approx(50.0)


def test_weighted_mean_access_level_invalido_lanza(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    with pytest.raises(ValueError):
        weighted_mean_access(access_df, population_df, level="pais", cfg=cfg)


# ---------------------------------------------------------------------------
# critical_gap_ranking
# ---------------------------------------------------------------------------

def test_critical_gap_ranking_excluye_unidades_sin_acceso_calculado(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    wma = weighted_mean_access(access_df, population_df, level="distrito", cfg=cfg)

    ranking = critical_gap_ranking(wma, n=5, poblacion_minima=0, cfg=cfg)
    # 010201 tiene t_min_medio_ponderado=NaN (dropna la excluye del ranking)
    assert len(ranking) == 1
    assert ranking.iloc[0]["nivel_id"] == "010101"
    assert ranking.iloc[0]["ranking"] == 1


def test_critical_gap_ranking_filtra_por_poblacion_minima(matrix_df, facility_resolutive_ids, population_df, cfg):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    wma = weighted_mean_access(access_df, population_df, level="distrito", cfg=cfg)

    ranking = critical_gap_ranking(wma, n=5, poblacion_minima=500, cfg=cfg)
    assert len(ranking) == 0  # la única unidad con acceso calculado tiene 400 de población < 500


# ---------------------------------------------------------------------------
# gini_access
# ---------------------------------------------------------------------------

def test_gini_access_calculo_a_mano(matrix_df, facility_resolutive_ids, population_df):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    resumen, lorenz = gini_access(access_df, population_df)

    # Solo d1 (t=10,w=100) y d2 (t=50,w=300) entran (d3 sin t_min se excluye).
    # Gini ponderado calculado a mano = 0.1875 (ver docstring de gini_access).
    assert resumen.iloc[0]["gini"] == pytest.approx(0.1875, abs=1e-6)
    assert resumen.iloc[0]["n_puntos"] == 2
    assert resumen.iloc[0]["poblacion_total_considerada"] == pytest.approx(400.0)
    assert resumen.iloc[0]["n_excluidos_sin_acceso"] == 1

    assert lorenz["pct_poblacion_acumulada"].iloc[0] == 0.0
    assert lorenz["pct_poblacion_acumulada"].iloc[-1] == pytest.approx(1.0)
    assert lorenz["pct_tiempo_acumulado"].iloc[-1] == pytest.approx(1.0)


def test_gini_access_cero_cuando_acceso_es_uniforme():
    access_df = pd.DataFrame({
        "id_demanda": ["a", "b", "c"],
        "t_min": [20.0, 20.0, 20.0],
        "flag_sin_acceso_enrutable": [False, False, False],
    })
    population_df = pd.DataFrame({"id": ["a", "b", "c"], "poblacion_censada": [10.0, 500.0, 30.0]})
    resumen, _ = gini_access(access_df, population_df)
    assert resumen.iloc[0]["gini"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# urban_rural_contrast
# ---------------------------------------------------------------------------

def test_urban_rural_contrast_requiere_clasificacion_previa(matrix_df, facility_resolutive_ids, population_df):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    with pytest.raises(KeyError):
        urban_rural_contrast(access_df, population_df)  # falta 'zona'


def test_urban_rural_contrast_medias_ponderadas(matrix_df, facility_resolutive_ids, population_df):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    clasificado = classify_urban_rural(population_df)
    out = urban_rural_contrast(access_df, clasificado).set_index("zona")

    assert out.loc["urbano", "t_min_medio_ponderado"] == pytest.approx(10.0)   # solo d1
    assert out.loc["urbano", "n_puntos"] == 1
    assert out.loc["rural", "t_min_medio_ponderado"] == pytest.approx(50.0)    # solo d2 tiene acceso
    assert out.loc["rural", "poblacion_sin_acceso_enrutable"] == pytest.approx(50.0)  # d3
    assert out.loc["rural", "n_puntos"] == 2


# ---------------------------------------------------------------------------
# access_vs_altitude
# ---------------------------------------------------------------------------

def test_access_vs_altitude_trae_z_y_t_min(matrix_df, facility_resolutive_ids, demanda_clean_like):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    por_punto, resumen = access_vs_altitude(access_df, demanda_clean_like)

    assert set(por_punto.columns) == {"id_demanda", "Z", "t_min"}
    assert por_punto.set_index("id_demanda").loc["d1", "Z"] == 100
    assert "nota_causalidad" in resumen.columns
    assert "causal" in resumen["nota_causalidad"].iloc[0].lower()


def test_access_vs_altitude_correlacion_monotona_perfecta():
    access_df = pd.DataFrame({
        "id_demanda": ["a", "b", "c", "d"],
        "t_min": [5.0, 15.0, 25.0, 35.0],
        "flag_sin_acceso_enrutable": [False] * 4,
    })
    demanda_clean = pd.DataFrame({"id": ["a", "b", "c", "d"], "Z": [100, 500, 1000, 2000]})
    _, resumen = access_vs_altitude(access_df, demanda_clean)
    assert resumen.iloc[0]["spearman_rho"] == pytest.approx(1.0)
    assert resumen.iloc[0]["n"] == 4


def test_access_vs_altitude_menos_de_3_puntos_no_calcula_correlacion(matrix_df, facility_resolutive_ids, demanda_clean_like):
    # Con la matriz sintética compartida, d3 no tiene t_min -> solo 2 puntos válidos.
    access_df = access_time(matrix_df, facility_resolutive_ids)
    _, resumen = access_vs_altitude(access_df, demanda_clean_like)
    assert pd.isna(resumen.iloc[0]["spearman_rho"])


# ---------------------------------------------------------------------------
# attach_sample_weights
# ---------------------------------------------------------------------------

def test_attach_sample_weights_bajo_el_limite_no_muestrea(cfg):
    demanda = pd.DataFrame({
        "id": ["d1", "d2", "d3"],
        "distrito": ["010101", "010101", "010201"],
    })
    out = attach_sample_weights(demanda, cfg)
    assert len(out) == 3
    assert (out["peso_muestral"] == 1.0).all()


# ---------------------------------------------------------------------------
# Indicadores del encabezado del panel (Fase 4)
# ---------------------------------------------------------------------------

def test_weighted_median_access_pondera_por_poblacion(matrix_df, facility_resolutive_ids, population_df):
    """d1 pesa 100 con t=10 y d2 pesa 300 con t=50 (d3 no tiene ruta y queda
    fuera). La mitad del peso son 200, que se alcanza recién dentro de d2:
    la mediana ponderada es 50, no 30 (el promedio simple de los dos)."""
    access_df = access_time(matrix_df, facility_resolutive_ids)
    assert weighted_median_access(access_df, population_df) == 50.0


def test_weighted_median_access_sin_poblacion_es_nan(matrix_df, facility_resolutive_ids, population_df):
    sin_poblacion = population_df.assign(poblacion_censada=np.nan)
    access_df = access_time(matrix_df, facility_resolutive_ids)
    assert np.isnan(weighted_median_access(access_df, sin_poblacion))


def test_population_within_separa_sin_ruta_de_lento(matrix_df, facility_resolutive_ids, population_df):
    """d3 (50 hab) no tiene ruta: no puede contar como 'por encima del
    umbral', porque su tiempo es desconocido, no alto."""
    access_df = access_time(matrix_df, facility_resolutive_ids)
    r = population_within(access_df, population_df, umbral_min=30)

    assert r["poblacion_bajo_umbral"] == 100.0    # d1, 10 min
    assert r["poblacion_sobre_umbral"] == 300.0   # d2, 50 min
    assert r["poblacion_sin_ruta"] == 50.0        # d3
    assert r["poblacion_total"] == 450.0
    assert r["pct_bajo_umbral"] == pytest.approx(22.22, abs=0.01)


def test_population_within_umbral_es_inclusivo(matrix_df, facility_resolutive_ids, population_df):
    """Un punto exactamente en el umbral cuenta como cubierto: '30 minutos o
    menos' es la lectura natural de la banda 0-30 de coverage_bands, y las
    dos definiciones tienen que coincidir."""
    access_df = access_time(matrix_df, facility_resolutive_ids)
    r = population_within(access_df, population_df, umbral_min=10)
    assert r["poblacion_bajo_umbral"] == 100.0


# ---------------------------------------------------------------------------
# Simulador de escenarios (Fase 4)
# ---------------------------------------------------------------------------

@pytest.fixture
def improvements_df():
    """f9 es un I-3 candidato a ascenso: deja a d2 a 20 min (hoy está a 50) y
    a d3 a 5 min (hoy no tiene ninguna ruta)."""
    return pd.DataFrame({
        "id_demanda": ["d2", "d3"],
        "id_candidato": ["f9", "f9"],
        "t_candidato_min": [20.0, 5.0],
        "fuente": ["estimado", "estimado"],
    })


def test_apply_upgrades_toma_el_minimo_y_marca_lo_mejorado(
    matrix_df, facility_resolutive_ids, improvements_df
):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    out = apply_upgrades(access_df, improvements_df, ["f9"]).set_index("id_demanda")

    assert out.loc["d1", "t_min"] == 10.0        # sin candidato cerca: no cambia
    assert not out.loc["d1", "mejorado"]
    assert out.loc["d2", "t_min"] == 20.0        # mejora de 50 a 20
    assert out.loc["d2", "mejorado"]
    assert out.loc["d3", "t_min"] == 5.0         # pasa de no tener ruta a tenerla
    assert out.loc["d3", "mejorado"]
    assert not out.loc["d3", "flag_sin_acceso_enrutable"]
    assert np.isnan(out.loc["d3", "t_min_original"])


def test_apply_upgrades_sin_seleccion_no_cambia_nada(matrix_df, facility_resolutive_ids, improvements_df):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    out = apply_upgrades(access_df, improvements_df, [])
    pd.testing.assert_series_equal(out["t_min"], access_df["t_min"], check_names=False)
    assert not out["mejorado"].any()


def test_apply_upgrades_ignora_candidatos_no_seleccionados(
    matrix_df, facility_resolutive_ids, improvements_df
):
    access_df = access_time(matrix_df, facility_resolutive_ids)
    out = apply_upgrades(access_df, improvements_df, ["OTRO_QUE_NO_ESTA"])
    assert not out["mejorado"].any()


def test_ranking_upgrade_gain_cuenta_solo_a_quien_cruza_el_umbral(
    matrix_df, facility_resolutive_ids, population_df, improvements_df
):
    """Con umbral de 30 min, ascender f9 mete dentro del umbral a d2 (300
    hab, hoy a 50 min) y a d3 (50 hab, hoy sin ruta). d1 ya estaba dentro y
    no suma."""
    access_df = access_time(matrix_df, facility_resolutive_ids)
    ranking = ranking_upgrade_gain(access_df, population_df, improvements_df, umbral_min=30)

    assert list(ranking["id_candidato"]) == ["f9"]
    assert ranking["poblacion_ganada"].iloc[0] == 350.0
    assert ranking["n_puntos_ganados"].iloc[0] == 2


def test_ranking_upgrade_gain_vacio_si_nadie_cruza(
    matrix_df, facility_resolutive_ids, population_df, improvements_df
):
    """Con un umbral de 2 minutos, ningún candidato acerca a nadie por debajo
    de esa marca: el ranking sale vacío en vez de con ganancias de cero."""
    access_df = access_time(matrix_df, facility_resolutive_ids)
    ranking = ranking_upgrade_gain(access_df, population_df, improvements_df, umbral_min=2)
    assert ranking.empty
