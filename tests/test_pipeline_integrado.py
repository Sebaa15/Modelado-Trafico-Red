"""Pruebas del Pipeline Integrado (Bayes -> Euler -> Kalman -> Decisiones)."""
import numpy as np
from modelos.kalman import FiltroKalmanCola
from escenarios.pipeline_integrado import ejecutar as ejecutar_pipeline


def test_filtro_kalman_cola_paso_y_horizonte():
    kf = FiltroKalmanCola(dt=1.0, Q=100.0, R=2500.0, x0=0.0, P0=100.0, B_max=50000.0)

    # Paso con entrada neta positiva (llegadas=1500, mu=1000 -> u_ctrl=+500)
    x_hat, P, K = kf.paso(u_ctrl=500.0, y_medida=510.0)
    assert x_hat > 0
    assert 0 <= x_hat <= 50000.0
    assert 0 < K < 1.0
    assert P > 0

    # Horizonte predictivo m=10
    x_fut, P_fut = kf.predecir_horizonte(u_ctrl_esperado=500.0, m_pasos=10)
    assert x_fut > x_hat
    assert P_fut > P
    assert x_fut <= 50000.0


def test_ejecucion_pipeline_integrado():
    res = ejecutar_pipeline()
    # Kalman debe reducir el RMSE de la medicion al menos un 50%
    assert res["rmse_kalman"] < res["rmse_medicion"] * 0.5
    # La alerta temprana debe preceder a la saturacion real
    assert res["primera_alerta"] is not None
    assert res["t_sat_real"] is not None
    assert res["primera_alerta"] < res["t_sat_real"]
