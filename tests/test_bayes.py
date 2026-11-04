"""Pruebas unitarias y de integracion para la Capa 2: Red Bayesiana de Contexto."""
import pytest
import numpy as np
from modelos.red_bayesiana import RedBayesianaTrafico, construir_red_bayesiana_pgmpy, inferir_pgmpy, PGMPY_DISPONIBLE


def test_inferencia_fase3_informe():
    """Verifica la consulta exacta de la Fase 3 del informe."""
    rb = RedBayesianaTrafico()
    res = rb.consulta_fase3()

    # Probabilidades de estados de llegada segun informe:
    # P(Lambda=Alta)=0.80, P(Lambda=Media)=0.15, P(Lambda=Baja)=0.05
    assert np.isclose(res["p_baja"], 0.05, atol=1e-3)
    assert np.isclose(res["p_media"], 0.15, atol=1e-3)
    assert np.isclose(res["p_alta"], 0.80, atol=1e-3)

    # Tasa esperada: 0.80*1500 + 0.15*1000 + 0.05*800 = 1390 pkt/s
    assert np.isclose(res["lambda_esperada_pps"], 1390.0, atol=1e-2)

    # Probabilidad de saturacion exacta: 0.8225 (82.25%)
    assert np.isclose(res["prob_saturacion"], 0.8225, atol=1e-4)
    assert np.isclose(res["prob_saturacion_pct"], 82.25, atol=1e-2)


def test_sensibilidad_contextual():
    """El horario pico con streaming debe presentar mayor riesgo que valle con web."""
    rb = RedBayesianaTrafico()
    _, p_sat_valle, _ = rb.inferir_contexto("Valle", "Web", "Bajo")
    _, p_sat_pico, _ = rb.inferir_contexto("Pico", "Streaming", "Alto")

    assert p_sat_valle < 0.10
    assert p_sat_pico > 0.85
    assert p_sat_pico > p_sat_valle * 8


def test_validacion_entradas_bayes():
    rb = RedBayesianaTrafico()
    with pytest.raises(ValueError):
        rb.inferir_contexto("Madrugada", "Streaming")
    with pytest.raises(ValueError):
        rb.inferir_contexto("Pico", "Juegos")
    with pytest.raises(ValueError):
        rb.inferir_contexto("Pico", "Web", historial="Extremo")


@pytest.mark.skipif(not PGMPY_DISPONIBLE, reason="pgmpy no esta instalado")
def test_modelo_pgmpy_coherente_con_directo():
    """Verifica que el modelo formal de pgmpy produzca el mismo resultado que el motor directo."""
    res_pgm = inferir_pgmpy({"H": "Pico", "A": "Streaming", "HC": "Moderado"})
    assert np.isclose(res_pgm["prob_saturacion"], 0.8225, atol=1e-4)
