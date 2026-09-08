"""Tests de routing.py que NO requieren un servidor OSRM corriendo: cubren
haversine, muestreo estratificado, nearest_facility y las comparaciones
entre perfiles usando matrices sintéticas.

Los tests que sí necesitan OSRM real (snap_points, compute_od_matrix contra
un servidor vivo) están marcados con @pytest.mark.osrm y se saltan si el
servidor no responde -- ver test_osrm_smoke() al final.
"""

import numpy as np
import pandas as pd
import pytest
import requests

from src.config_loader import load_config
from src.routing import (
    haversine_km, sample_demand_points, nearest_facility,
    compare_walk_vs_drive, compare_all_modes, validate_deviation_factor,
    apply_fallback, OSRMClient,
)


def test_haversine_known_distance():
    # Lima - Piura, distancia real por línea recta ~860 km
    d = haversine_km(np.array([-77.03]), np.array([-12.05]), np.array([-80.63]), np.array([-5.19]))
    assert 830 < d[0] < 890


def test_sample_demand_points_under_limit_returns_unchanged():
    cfg = load_config()
    df = pd.DataFrame({"id": range(10), "distrito": ["D1"] * 10})
    out = sample_demand_points(df, col_distrito="distrito", col_poblacion=None, cfg=cfg)
    assert len(out) == 10
    assert (out["peso_muestral"] == 1.0).all()


def test_sample_demand_points_over_limit_is_capped():
    cfg = load_config()
    cfg = {**cfg, "enrutamiento": {**cfg["enrutamiento"],
           "muestreo": {**cfg["enrutamiento"]["muestreo"], "limite_puntos_demanda": 50}}}
    df = pd.DataFrame({"id": range(500), "distrito": np.random.choice(["D1", "D2", "D3"], 500)})
    out = sample_demand_points(df, col_distrito="distrito", col_poblacion=None, cfg=cfg)
    assert len(out) <= 60  # margen por redondeo de round() en cada grupo


def test_nearest_facility_picks_min_duration():
    m = pd.DataFrame({
        "id_demanda": ["D1", "D1", "D2"],
        "id_instalacion": ["F1", "F2", "F1"],
        "duracion_min": [12.0, 30.0, 5.0],
        "distancia_km": [5.0, 15.0, 2.0],
        "enrutable": [True, True, True],
    })
    out = nearest_facility(m, facility_resolutive_ids={"F1", "F2"})
    assert out.set_index("id_demanda").loc["D1", "id_instalacion"] == "F1"


def test_compare_walk_vs_drive_flags_different_facility():
    car = pd.DataFrame({
        "id_demanda": ["D1"], "id_instalacion": ["F1"], "duracion_min": [10.0],
        "distancia_km": [4.0], "enrutable": [True],
    })
    foot = pd.DataFrame({
        "id_demanda": ["D1"], "id_instalacion": ["F2"], "duracion_min": [180.0],
        "distancia_km": [12.0], "enrutable": [True],
    })
    comp = compare_walk_vs_drive(car, foot, facility_resolutive_ids={"F1", "F2"})
    assert comp.iloc[0]["misma_instalacion"] == False  # noqa: E712
    assert comp.iloc[0]["ratio_foot_car"] == pytest.approx(18.0)


def test_validate_deviation_factor_reasonable_range():
    matrix = pd.DataFrame({
        "id_demanda": ["D1"], "id_instalacion": ["F1"],
        "duracion_min": [20.0], "distancia_km": [12.0], "enrutable": [True],
    })
    demand = pd.DataFrame({"id": ["D1"], "lon": [-77.0], "lat": [-12.0]})
    facility = pd.DataFrame({"id": ["F1"], "lon": [-77.08], "lat": [-12.06]})
    resumen = validate_deviation_factor(matrix, demand, facility, "id", "id")
    # distancia recta Lima-ish entre esos dos puntos es unos ~9-10km, red=12km -> ratio > 1
    assert resumen["factor_mediano"] > 1.0


def test_apply_fallback_marks_estimated_rows():
    nearest = pd.DataFrame({
        "id_demanda": ["D1"], "id_instalacion": ["F1"], "duracion_min": [10.0], "distancia_km": [4.0],
    })
    demand = pd.DataFrame({"id": ["D1", "D2"], "lon": [-77.0, -77.5], "lat": [-12.0, -12.3]})
    facility = pd.DataFrame({"id": ["F1"], "lon": [-77.08], "lat": [-12.06]})
    out = apply_fallback(nearest, demand, facility, unroutable_demand_ids=["D2"],
                          col_id_demand="id", col_id_facility="id", factor_desvio=1.35)
    fila_d2 = out[out["id_demanda"] == "D2"].iloc[0]
    assert fila_d2["estimado_por_fallback"] == True  # noqa: E712
    assert fila_d2["distancia_km"] > 0


@pytest.mark.osrm
def test_osrm_smoke():
    """Se salta automáticamente si no hay un servidor OSRM 'car' corriendo
    en el puerto de config.md (ver docker/docker-compose.yml)."""
    cfg = load_config()
    port = cfg["enrutamiento"]["osrm"]["puertos"]["car"]
    try:
        requests.get(f"http://localhost:{port}/nearest/v1/car/-77.03,-12.05", timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip("No hay servidor OSRM corriendo en localhost -- levantar con docker compose.")

    client = OSRMClient("car", cfg)
    res = client.nearest(-77.03, -12.05)
    assert res is not None
    assert res["snap_distance_m"] >= 0
