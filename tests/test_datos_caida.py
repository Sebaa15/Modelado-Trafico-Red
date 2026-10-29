"""Pruebas unitarias de integridad para el dataset CAIDA y homogeneizacion."""
import numpy as np
import pytest

from datos.cargador_caida import (
    cargar_dataset_caida,
    resumen_estadistico_caida,
    generar_serie_temporal_caida,
    generar_banco_escenarios_homogeneo
)
from modelos.config import (
    DURACION_ESTANDAR,
    BUFFER_MAX,
    CAPACIDAD_NOMINAL,
    CAPACIDAD_DEGRADADA
)


def test_carga_dataset_caida():
    df = cargar_dataset_caida()
    assert len(df) == 362
    columnas_esperadas = {"monitor", "date", "time_utc", "link_load_pct", "pkts_per_sec", "mean_pkt_size_bytes"}
    assert columnas_esperadas.issubset(set(df.columns))
    assert np.all(df["pkts_per_sec"] > 0)


def test_resumen_estadistico_caida():
    stats = resumen_estadistico_caida()
    assert stats["total_registros"] == 362
    assert "chicago-dirA" in stats["monitores"]
    assert stats["tasa_paquetes_media_pps"] > 1000
    assert stats["carga_media_enlace_pct"] > 0


def test_generador_serie_temporal_caida():
    tiempo, paquetes, flujos = generar_serie_temporal_caida(duracion=300, dt=1.0, semilla=42)
    assert len(tiempo) == 300
    assert len(paquetes) == 300
    assert len(flujos) == 300
    assert np.all(paquetes > 0)
    assert np.all(flujos > 0)


def test_banco_escenarios_homogeneo():
    banco = generar_banco_escenarios_homogeneo(duracion=DURACION_ESTANDAR, dt=1.0)
    assert "tiempo" in banco
    assert len(banco["tiempo"]) == DURACION_ESTANDAR

    for esc_id in ["A", "B", "C", "D", "E"]:
        assert esc_id in banco
        esc = banco[esc_id]
        assert len(esc["llegadas"]) == DURACION_ESTANDAR
        assert esc["buffer_max"] == BUFFER_MAX
        assert esc["capacidad"] in (CAPACIDAD_NOMINAL, CAPACIDAD_DEGRADADA, 1600.0)
