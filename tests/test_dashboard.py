"""Tests de la Fase 4 (panel).

Dos bloques:

1. Unitarios sobre `src/dashboard_data.py`: DataFrames sintéticos, sin tocar
   disco ni Streamlit (mismo estilo que test_metrics.py). Lo que se prueba
   aquí es el contrato con metrics.py, que es donde un panel se rompe en
   silencio: si `split_access_population` dejara pasar a los puntos no
   muestreados, todas las cifras del panel bajarían sin que nada falle.

2. De integración, marcados `artefactos`: requieren haber corrido
   `python run_fase4.py`. Verifican lo único que de verdad importa del
   panel -- que muestre los MISMOS números que el informe -- y que la app
   levante sin excepciones.

    pytest tests/test_dashboard.py -v -m "not artefactos"   # sin artefactos
    pytest tests/test_dashboard.py -v                        # todo
"""

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.dashboard_data import (
    PUNTOS_DASHBOARD, artefacto, assign_bands, band_labels, etiquetas_legibles,
    filter_instalaciones, filter_puntos, formato_minutos, orden_bandas,
    split_access_population,
)
from src.metrics import coverage_bands


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def puntos():
    """5 centros poblados: 3 en la muestra enrutada (uno de ellos sin ruta a
    ninguna instalación resolutiva) y 2 fuera de la muestra."""
    return pd.DataFrame({
        "id": ["d1", "d2", "d3", "d4", "d5"],
        "DEP": ["PIURA", "PIURA", "LORETO", "LORETO", "AYACUCHO"],
        "PROV": ["P1", "P1", "P2", "P2", "P3"],
        "DIST": ["D1", "D1", "D2", "D2", "D3"],
        "distrito": ["200101", "200101", "160101", "160101", "050101"],
        "zona": ["urbano", "rural", "rural", "rural", "rural"],
        "poblacion_censada": [100.0, 300.0, 50.0, 700.0, 20.0],
        "peso_muestral": [1.0, 1.0, 1.0, np.nan, np.nan],
        "Z": [10, 200, 120, 130, 3000],
        "lon": [-80.0, -80.1, -74.0, -74.1, -74.2],
        "lat": [-5.0, -5.1, -3.0, -3.1, -13.0],
        "t_min": [10.0, 200.0, np.nan, np.nan, np.nan],
        "id_instalacion": ["f1", "f1", None, None, None],
        "flag_sin_acceso_enrutable": [False, False, True, False, False],
        "en_muestra_enrutada": [True, True, True, False, False],
        "banda": ["0-30", ">120", "sin_acceso_enrutable", "no_muestreado", "no_muestreado"],
    })


# ---------------------------------------------------------------------------
# split_access_population
# ---------------------------------------------------------------------------

def test_split_access_solo_incluye_la_muestra_enrutada(puntos):
    access_df, population_df = split_access_population(puntos)
    assert list(access_df["id_demanda"]) == ["d1", "d2", "d3"]
    # El universo poblacional NO se recorta: los puntos fuera de la muestra
    # siguen ahí con peso_muestral NaN (peso efectivo 0), igual que en Fase 3.
    assert len(population_df) == 5


def test_split_no_deja_columnas_de_acceso_en_population(puntos):
    _, population_df = split_access_population(puntos)
    for col in ["t_min", "flag_sin_acceso_enrutable", "id_instalacion", "en_muestra_enrutada", "banda"]:
        assert col not in population_df.columns, f"{col} contaminaría el merge de metrics._join_by_id"


def test_split_conserva_el_flag_de_sin_acceso(puntos):
    access_df, _ = split_access_population(puntos)
    assert access_df.set_index("id_demanda").loc["d3", "flag_sin_acceso_enrutable"]
    assert not access_df.set_index("id_demanda").loc["d1", "flag_sin_acceso_enrutable"]


def test_split_falla_claro_si_faltan_columnas():
    with pytest.raises(KeyError, match="faltan columnas"):
        split_access_population(pd.DataFrame({"id": ["d1"]}))


# ---------------------------------------------------------------------------
# Bandas
# ---------------------------------------------------------------------------

def test_band_labels_salen_de_config(cfg):
    bordes = cfg["metricas"]["bandas_acceso_min"]
    assert band_labels(cfg) == ["0-30", "30-60", "60-120", ">120"]
    assert orden_bandas(cfg)[-1] == "sin_acceso_enrutable"
    assert len(band_labels(cfg)) == len(bordes) + 1


def test_assign_bands_bordes_y_nan(cfg):
    serie = pd.Series([0.0, 30.0, 30.1, 60.0, 120.0, 120.1, np.nan])
    bandas = assign_bands(serie, cfg)
    # pd.cut cierra por la derecha: 30 min cae en "0-30", no en "30-60".
    assert list(bandas[:6]) == ["0-30", "0-30", "30-60", "30-60", "60-120", ">120"]
    assert bandas.iloc[-1] == "sin_acceso_enrutable"


def test_assign_bands_no_contradice_a_coverage_bands(puntos, cfg):
    """El mapa colorea con assign_bands y la tabla se calcula con
    metrics.coverage_bands. Si un día alguien cambia los bordes en un sitio y
    no en el otro, el panel mostraría un punto rojo dentro de una banda verde.
    Con población unitaria, el reparto de puntos debe coincidir."""
    access_df, population_df = split_access_population(puntos)
    population_df = population_df.assign(poblacion_censada=1.0, peso_muestral=1.0)

    cb = coverage_bands(access_df, population_df, cfg).set_index("banda")["poblacion"]
    directo = assign_bands(access_df["t_min"], cfg).value_counts()

    for banda, n in directo.items():
        assert cb.get(banda, 0.0) == pytest.approx(float(n)), f"discrepancia en la banda {banda}"


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def test_filter_puntos_none_es_sin_filtro_y_lista_vacia_filtra_todo(puntos):
    assert len(filter_puntos(puntos)) == 5
    assert len(filter_puntos(puntos, departamentos=[])) == 0


def test_filter_puntos_por_departamento_y_zona(puntos):
    assert set(filter_puntos(puntos, departamentos=["LORETO"])["id"]) == {"d3", "d4"}
    assert set(filter_puntos(puntos, zonas=["urbano"])["id"]) == {"d1"}
    assert len(filter_puntos(puntos, departamentos=["PIURA"], zonas=["rural"])) == 1


def test_filter_puntos_solo_muestra_enrutada(puntos):
    assert len(filter_puntos(puntos, solo_muestra_enrutada=True)) == 3


def test_filter_puntos_por_provincia(puntos):
    assert set(filter_puntos(puntos, provincias=["P2"])["id"]) == {"d3", "d4"}
    # El filtro de departamento se aplica primero, así que una provincia
    # homónima de otro departamento no se cuela.
    assert len(filter_puntos(puntos, departamentos=["PIURA"], provincias=["P2"])) == 0


@pytest.fixture
def instalaciones():
    return pd.DataFrame({
        "COD_IPRESS": ["1", "2", "3", "4"],
        "DEPARTAMENTO": ["PIURA", "PIURA", "LORETO", "LORETO"],
        "PROVINCIA": ["PIURA", "SULLANA", "MAYNAS", "MAYNAS"],
        "categoria_norm": ["II-1", "I-3", "I-4", "SIN_CATEGORIA"],
        "INSTITUCION": ["GOBIERNO REGIONAL", "PRIVADO", "ESSALUD", "PRIVADO"],
        "es_resolutivo": [True, False, False, False],
    })


def test_filter_instalaciones_solo_resolutivas(instalaciones):
    assert len(filter_instalaciones(instalaciones, solo_resolutivas=True)) == 1
    assert len(filter_instalaciones(instalaciones, departamentos=["LORETO"],
                                     solo_resolutivas=True)) == 0


def test_filter_instalaciones_por_categoria_e_institucion(instalaciones):
    assert set(filter_instalaciones(instalaciones, categorias=["I-3", "I-4"])["COD_IPRESS"]) == {"2", "3"}
    assert set(filter_instalaciones(instalaciones, instituciones=["PRIVADO"])["COD_IPRESS"]) == {"2", "4"}
    assert len(filter_instalaciones(instalaciones, categorias=["I-3"], instituciones=["ESSALUD"])) == 0


def test_filter_instalaciones_sin_categoria_es_una_categoria_mas(instalaciones):
    """SIN_CATEGORIA no es un nulo que se filtre solo: es un valor que el
    usuario puede seleccionar, porque esos 390 establecimientos existen y su
    ausencia de categoría es un dato."""
    assert len(filter_instalaciones(instalaciones, categorias=["SIN_CATEGORIA"])) == 1


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valor,esperado", [
    (45.0, "45 min"), (89.9, "90 min"), (120.0, "2.0 h"),
    (2880.0, "2.0 días"), (np.nan, "s/d"), (None, "s/d"),
])
def test_formato_minutos(valor, esperado):
    assert formato_minutos(valor) == esperado


def test_etiquetas_legibles_no_toca_columnas_desconocidas():
    df = pd.DataFrame({"nivel_nombre": ["A"], "columna_rara": [1]})
    out = etiquetas_legibles(df)
    assert list(out.columns) == ["Unidad", "columna_rara"]


# ---------------------------------------------------------------------------
# Integración: el panel muestra los mismos números que el informe
# ---------------------------------------------------------------------------

requiere_artefactos = pytest.mark.skipif(
    not artefacto(PUNTOS_DASHBOARD).exists(),
    reason="Faltan los artefactos de Fase 4; corre `python run_fase4.py`.",
)


@pytest.mark.artefactos
@requiere_artefactos
def test_panel_sin_filtros_reproduce_coverage_bands_del_informe(cfg):
    """Sin filtros, recalcular desde los artefactos del panel tiene que dar
    exactamente `data/outputs/coverage_bands.csv`, que es la tabla que cita
    el informe. Este test es el que evita que panel e informe se separen."""
    from src.dashboard_data import load_output_csv, load_puntos

    access_df, population_df = split_access_population(load_puntos(cfg))
    recalculado = coverage_bands(access_df, population_df, cfg)
    esperado = load_output_csv("coverage_bands", cfg)

    pd.testing.assert_frame_equal(
        recalculado.reset_index(drop=True), esperado.reset_index(drop=True),
        check_dtype=False, atol=1e-6,
    )


@pytest.mark.artefactos
@requiere_artefactos
def test_app_levanta_sin_excepciones():
    """Smoke test de la app completa con el runner oficial de Streamlit:
    ejecuta el script entero (las 8 pestañas se renderizan siempre) y luego
    cambia un filtro para forzar el recálculo."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    at = AppTest.from_file(str(artefacto("app.py")), default_timeout=600).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.metric) > 0

    at.sidebar.multiselect[0].set_value(["LORETO"]).run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.artefactos
@requiere_artefactos
def test_mejoras_candidatos_solo_contiene_pares_que_mejoran():
    """Invariante del artefacto del simulador: si guardara pares que NO
    mejoran, apply_upgrades seguiría dando el resultado correcto (toma un
    mínimo), pero el archivo pesaría cuatro veces más y el ranking de
    ganancia individual recorrería basura."""
    from src.dashboard_data import load_mejoras_candidatos, load_puntos

    cfg = load_config()
    mejoras = load_mejoras_candidatos(cfg)
    actual = load_puntos(cfg).set_index("id")["t_min"]

    t_actual = mejoras["id_demanda"].map(actual)
    sin_ruta_hoy = t_actual.isna()
    mejora_de_verdad = mejoras["t_candidato_min"] < t_actual

    assert (sin_ruta_hoy | mejora_de_verdad).all()
    assert mejoras["t_candidato_min"].gt(0).all()
    assert set(mejoras["fuente"]).issubset({"estimado", "osrm"})
