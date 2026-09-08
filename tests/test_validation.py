"""Tests de la capa de validación. No requieren red ni OSRM: son puramente
sobre DataFrames sintéticos, así se pueden correr en CI sin infraestructura."""

import pandas as pd
import pytest

from src.config_loader import load_config
from src.validation import (
    normalize_category, is_active, is_resolutive,
    flag_missing_coords, flag_out_of_bbox, flag_lat_lon_swapped,
    flag_duplicate_codes, run_quality_pipeline, apply_category_rules,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_normalize_category_variants(cfg):
    assert normalize_category("II-1", cfg) == "II-1"
    assert normalize_category("ii - 1", cfg) == "II-1"
    assert normalize_category("CATEGORIA III-E", cfg) == "III-E"
    assert normalize_category("no existe esto", cfg) == "SIN_CATEGORIA"


def test_is_resolutive_requires_active_and_category(cfg):
    assert is_resolutive("II-1", "EN FUNCIONAMIENTO", cfg) is True
    assert is_resolutive("II-1", "CERRADO", cfg) is False       # activo pero categoría I -> no resolutivo
    assert is_resolutive("I-1", "EN FUNCIONAMIENTO", cfg) is False


def test_flag_missing_coords():
    df = pd.DataFrame({"lon": [-80.0, None, 0.0, -75.0], "lat": [-5.0, -5.0, -5.0, None]})
    flags = flag_missing_coords(df, "lon", "lat")
    assert flags.tolist() == [False, True, True, True]


def test_flag_out_of_bbox(cfg):
    df = pd.DataFrame({"lon": [-75.0, 40.0], "lat": [-10.0, -10.0]})
    flags = flag_out_of_bbox(df, "lon", "lat", cfg)
    assert flags.tolist() == [False, True]


def test_flag_lat_lon_swapped(cfg):
    # -0.06 en la posición de "lon" y -80.6 en "lat" -> claramente invertido
    df = pd.DataFrame({"lon": [-0.06], "lat": [-80.6]})
    flags = flag_lat_lon_swapped(df, "lon", "lat", cfg)
    assert flags.iloc[0] == True  # noqa: E712


def test_flag_duplicate_codes():
    df = pd.DataFrame({"codigo": ["A", "B", "A", "C"]})
    flags = flag_duplicate_codes(df, "codigo")
    assert flags.tolist() == [False, False, True, False]


def test_run_quality_pipeline_end_to_end(cfg):
    df = pd.DataFrame({
        "codigo": ["A1", "A2", "A3"],
        "lon": [-80.6, -0.06, None],
        "lat": [-5.2, -80.6, -5.0],
        "ubigeo": ["200101", "200101", "160101"],
        "categoria_ipress": ["II-1", "I-1", "II-2"],
        "estado": ["EN FUNCIONAMIENTO"] * 3,
        "nombre": ["A", "B", "C"],
    })
    df_cat = apply_category_rules(df, "categoria_ipress", "estado", cfg)
    clean, report = run_quality_pipeline(
        df_cat, col_lon="lon", col_lat="lat", col_codigo="codigo",
        col_ubigeo="ubigeo", col_texto_libre=["nombre"], cfg=cfg,
    )
    # A3 (coords faltantes) debe salir del df limpio
    assert "A3" not in clean["codigo"].values
    # A2 debe seguir presente y con coords corregidas (swap)
    fila_a2 = clean[clean["codigo"] == "A2"].iloc[0]
    assert fila_a2["lon"] == pytest.approx(-80.6)
    assert fila_a2["lat"] == pytest.approx(-0.06)
    assert set(report["regla"]) == {
        "coords_faltantes", "fuera_de_bbox_peru", "lat_lon_intercambiados",
        "fuera_de_su_propio_distrito", "codigos_duplicados", "encoding_utf8_vs_latin1",
    }
