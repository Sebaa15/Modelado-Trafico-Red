import numpy as np

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from modelos.kalman_extendido import estimar_cola_ekf


def preparar_trafico():
    n = 1200
    normal = np.full(n, 800.0)
    pico = normal.copy()
    pico[200:800] = 1500.0
    return normal, pico


def test_escenarios_euler():
    normal, pico = preparar_trafico()

    a = modelo_euler(normal, 1000.0)
    b = modelo_euler(pico, 1000.0)
    c = modelo_euler(normal, 600.0)
    limitado = modelo_euler(
        pico, 1000.0, buffer_max=100000.0
    )

    assert np.max(a["cola"]) == 0
    assert np.max(b["cola"]) == 300000
    assert b["cola"][-1] == 220000
    assert np.max(c["cola"]) == 240000

    assert np.max(limitado["cola"]) == 100000
    assert np.sum(limitado["perdidas"]) == 200000

    for resultado in (a, b, c, limitado):
        assert len(resultado["cola"]) == 1201
        assert len(resultado["perdidas"]) == 1200
        assert np.all(resultado["cola"] >= 0)


def test_integracion_euler_kalman():
    normal, pico = preparar_trafico()

    euler = modelo_euler(
        pico, 1000.0, buffer_max=100000.0
    )

    real = euler["cola"]

    rng = np.random.default_rng(42)
    mediciones = real + rng.normal(
        0, 50, len(real)
    )

    estimado, sigma = estimar_cola(
        pico, mediciones, 1000.0,
        Q=100, R=2500, buffer_max=100000.0
    )

    assert len(estimado) == len(real)
    assert np.all(np.isfinite(estimado))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)
    assert np.all(estimado >= 0)
    assert np.all(estimado <= 100000)

    mae_medicion = np.mean(np.abs(mediciones - real))
    mae_kalman = np.mean(np.abs(estimado - real))

    assert mae_kalman < mae_medicion


def test_integracion_euler_ekf():
    normal, pico = preparar_trafico()

    euler = modelo_euler(
        pico, 1000.0, buffer_max=100000.0
    )

    real = euler["cola"]

    rng = np.random.default_rng(42)
    mediciones = real + rng.normal(
        0, 50, len(real)
    )

    estimado, sigma = estimar_cola_ekf(
        pico, mediciones, 1000.0,
        Q=100, R=2500, buffer_max=100000.0
    )

    assert len(estimado) == len(real)
    assert np.all(np.isfinite(estimado))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)
    assert np.all(estimado >= 0)
    assert np.all(estimado <= 100000)
