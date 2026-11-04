import numpy as np
from escenarios.validacion_independiente import generar_experimento, rmse_prueba


def test_serie_sintetica_reproducible():
    a = generar_experimento(15)
    b = generar_experimento(15)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a[1]) == len(a[0]) + 1
    assert np.all(a[1] >= 0)


def test_metrica_filtro_finita():
    valor = rmse_prueba(64, 2500, [12, 13])
    assert np.isfinite(valor) and valor >= 0
