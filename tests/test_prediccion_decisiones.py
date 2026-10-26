import numpy as np
import pytest

from modelos.prediccion import predecir_cola, primer_instante_buffer_lleno
from modelos.decisiones import comparar_capacidades, evaluar_congestion


def test_llegadas_elevadas_agotan_buffer_y_generan_perdidas():
    perfil = np.full(120, 1500.0)
    pronostico = predecir_cola(75000.0, perfil, 1000.0, buffer_max=100000.0)
    assert len(pronostico["cola"]) == 121
    assert pronostico["cola"][-1] == 100000
    assert np.sum(pronostico["perdidas"]) == 35000
    assert primer_instante_buffer_lleno(pronostico) == 50.0
    assert evaluar_congestion(pronostico)["nivel"] == "CRITICA"


def test_ampliar_capacidad_evitaria_perdidas_bajo_este_perfil():
    perfil = np.full(120, 1500.0)
    comparacion = comparar_capacidades(
        75000.0, perfil, capacidad_actual=1000.0,
        nueva_capacidad=1500.0, buffer_max=100000.0
    )
    assert np.sum(comparacion["actual"]["perdidas"]) == 35000
    assert np.sum(comparacion["alternativa"]["perdidas"]) == 0
    assert comparacion["alternativa"]["cola"][-1] == 75000


def test_sin_buffer_no_inventa_perdidas():
    pred = predecir_cola(0, np.full(10, 1500), 1000)
    assert pred["buffer_max"] is None
    assert np.sum(pred["perdidas"]) == 0
    assert evaluar_congestion(pred)["nivel"] == "PREVENTIVA"


def test_rechazo_entradas_invalidas():
    with pytest.raises(ValueError):
        predecir_cola(-1, [1000], 1000)
    with pytest.raises(ValueError):
        predecir_cola(0, [], 1000)
    with pytest.raises(ValueError):
        predecir_cola(100001, [1000], 1000, buffer_max=100000)
