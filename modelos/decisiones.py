"""Alertas de congestion y simulacion de alternativas de capacidad.

Estas recomendaciones se basan en escenarios, no son politicas RED
ni garantias de desempeno de una red real.
"""
import numpy as np
from modelos.prediccion import predecir_cola, primer_instante_buffer_lleno


def evaluar_congestion(pronostico, umbral_carga=0.90,
                        umbral_buffer=0.80):
    if not 0 < umbral_carga <= 1:
        raise ValueError("umbral_carga debe estar entre 0 y 1.")
    if not 0 < umbral_buffer <= 1:
        raise ValueError("umbral_buffer debe estar entre 0 y 1.")

    ofrecida = np.asarray(pronostico["utilizacion"], dtype=float)
    cola = np.asarray(pronostico["cola"], dtype=float)
    perdidas = np.asarray(pronostico["perdidas"], dtype=float)
    limite = pronostico["buffer_max"]
    pico_carga = float(np.max(ofrecida))
    perdida_total = float(np.sum(perdidas))
    fraccion_max_buffer = (None if limite is None else float(np.max(cola) / limite))
    tiempo_lleno = primer_instante_buffer_lleno(pronostico)

    if perdida_total > 0 or tiempo_lleno is not None:
        nivel = "CRITICA"
        razon = "El buffer se llena o aparecen perdidas dentro del horizonte."
    elif fraccion_max_buffer is not None and fraccion_max_buffer >= umbral_buffer:
        nivel = "ALTA"
        razon = "La cola alcanza un porcentaje elevado del buffer."
    elif pico_carga > 1 or pico_carga >= umbral_carga:
        nivel = "PREVENTIVA"
        razon = "La carga ofrecida se acerca o supera la capacidad de servicio."
    else:
        nivel = "NORMAL"
        razon = "No se detectan umbrales de alerta en el horizonte supuesto."

    return {
        "nivel": nivel, "razon": razon,
        "carga_maxima_pct": 100 * pico_carga,
        "ocupacion_maxima_pct": None if fraccion_max_buffer is None else 100 * fraccion_max_buffer,
        "paquetes_perdidos": perdida_total,
        "tiempo_buffer_lleno_s": tiempo_lleno,
        "cola_final": float(cola[-1]),
    }


def comparar_capacidades(cola_estimada, llegadas_futuras, capacidad_actual,
                         nueva_capacidad, dt=1.0, buffer_max=None):
    if nueva_capacidad <= 0:
        raise ValueError("La capacidad propuesta debe ser positiva.")
    base = predecir_cola(cola_estimada, llegadas_futuras,
                        capacidad_actual, dt, buffer_max)
    alternativa = predecir_cola(cola_estimada, llegadas_futuras,
                               nueva_capacidad, dt, buffer_max)
    return {"actual": base, "alternativa": alternativa,
            "diagnostico_actual": evaluar_congestion(base),
            "diagnostico_alternativo": evaluar_congestion(alternativa)}
