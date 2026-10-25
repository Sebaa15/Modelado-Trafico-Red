import numpy as np

def modelo_euler(llegadas, capacidad, dt=1.0,
                 cola_inicial=0.0, buffer_max=None):

    llegadas = np.asarray(llegadas, dtype=float)

    # Validar entradas
    if llegadas.ndim != 1 or len(llegadas) == 0:
        raise ValueError("Las llegadas deben ser una serie no vacia.")

    if not np.all(np.isfinite(llegadas)) or np.any(llegadas < 0):
        raise ValueError("Las llegadas deben ser finitas y no negativas.")

    if not np.isfinite(capacidad) or capacidad <= 0:
        raise ValueError("La capacidad debe ser positiva.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("El intervalo dt debe ser positivo.")

    if not np.isfinite(cola_inicial) or cola_inicial < 0:
        raise ValueError("La cola inicial no puede ser negativa.")

    if buffer_max is not None:
        if not np.isfinite(buffer_max) or buffer_max <= 0:
            raise ValueError("El buffer debe ser positivo.")

        if cola_inicial > buffer_max:
            raise ValueError("La cola inicial supera el buffer.")

    n = len(llegadas)

    tiempo = np.arange(n + 1) * dt
    cola = np.zeros(n + 1)
    utilizacion = llegadas / capacidad
    perdidas = np.zeros(n)

    cola[0] = cola_inicial

    for k in range(n):

        # Euler: actualizar la cola
        cola_propuesta = max(
            0.0,
            cola[k] + dt * (llegadas[k] - capacidad)
        )

        # Aplicar el limite del buffer si existe
        if buffer_max is not None:
            perdidas[k] = max(
                0.0,
                cola_propuesta - buffer_max
            )

            cola[k + 1] = min(
                cola_propuesta,
                buffer_max
            )
        else:
            cola[k + 1] = cola_propuesta

    resu = {
        "tiempo": tiempo,
        "cola": cola,
        "utilizacion": utilizacion,
        "llegadas": llegadas,
        "perdidas": perdidas,
        "capacidad": capacidad,
        "dt": dt,
        "buffer_max": buffer_max
    }

    return resu
