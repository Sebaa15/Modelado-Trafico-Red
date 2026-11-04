"""Pruebas del Escenario E (Traza CAIDA, Micro-rafagas y Flujos como Leading Indicator)."""
import numpy as np
from escenarios.escenario_e import generar_serie_caida, ejecutar as ejecutar_escenario_e


def test_generador_serie_caida():
    tiempo, lambdas, flujos = generar_serie_caida(duracion=100, dt=1.0, semilla=42)
    assert len(tiempo) == 100
    assert len(lambdas) == 100
    assert len(flujos) == 100
    assert np.all(lambdas > 0)
    assert np.all(flujos > 0)
    # Debe ser reproducible con la misma semilla
    t2, l2, f2 = generar_serie_caida(duracion=100, dt=1.0, semilla=42)
    assert np.allclose(lambdas, l2)
    assert np.allclose(flujos, f2)


def test_ejecucion_escenario_e():
    res = ejecutar_escenario_e()
    assert res["mu_calibrada"] > 1000.0
    assert res["cola_max"] <= 50000.0
    assert res["rmse_kalman"] < 500.0
    # La alerta con flujos debe existir y ser previa a la saturacion real
    assert res["primera_alerta_flujos"] is not None
    assert res["t_sat_real"] is not None
    assert res["primera_alerta_flujos"] < res["t_sat_real"]
