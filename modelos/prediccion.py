"""Pronostico condicional a un perfil supuesto de llegadas futuras.

La simulacion parte del ultimo estado estimado por Kalman, aplica el
modelo Euler de la cola y devuelve trayectorias y perdidas previstas.
No identifica por si sola el trafico futuro ni usa informacion de Bayes.
"""
import numpy as np
from euler.euler import modelo_euler


def predecir_cola(cola_estimada, llegadas_futuras, capacidad,
                  dt=1.0, buffer_max=None):
    if not np.isfinite(cola_estimada) or cola_estimada < 0:
        raise ValueError("Cola estimada inicial invalida.")
    if buffer_max is not None and cola_estimada > buffer_max:
        raise ValueError("La cola estimada supera el buffer.")
    llegadas_futuras = np.asarray(llegadas_futuras, dtype=float)
    if llegadas_futuras.ndim != 1 or len(llegadas_futuras) == 0:
        raise ValueError("Indicar al menos una llegada futura.")
    return modelo_euler(llegadas_futuras, capacidad,
                        dt=dt, cola_inicial=float(cola_estimada),
                        buffer_max=buffer_max)


def primer_instante_buffer_lleno(pronostico):
    """Tiempo relativo (en segundos) al primer estado de buffer lleno."""
    buffer_max = pronostico["buffer_max"]
    if buffer_max is None:
        return None
    indices = np.flatnonzero(
        np.asarray(pronostico["cola"]) >= buffer_max - 1e-9
    )
    return None if len(indices) == 0 else float(pronostico["tiempo"][indices[0]])
