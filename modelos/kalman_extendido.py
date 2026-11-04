import numpy as np


def estimar_cola_ekf(llegadas, mediciones, mu,
                     dt=1.0, Q=100.0, R=2500.0,
                     buffer_max=None):

    llegadas = np.asarray(llegadas, dtype=float)
    mediciones = np.asarray(mediciones, dtype=float)

    if llegadas.ndim != 1 or mediciones.ndim != 1:
        raise ValueError("Las entradas deben ser vectores.")

    if len(llegadas) == 0 or len(mediciones) != len(llegadas) + 1:
        raise ValueError("Se esperan N llegadas y N+1 mediciones.")

    if not np.all(np.isfinite(llegadas)) or np.any(llegadas < 0):
        raise ValueError("Llegadas invalidas.")

    if not np.all(np.isfinite(mediciones)):
        raise ValueError("Mediciones invalidas.")

    if mu <= 0 or dt <= 0 or Q < 0 or R <= 0:
        raise ValueError("Parametros invalidos.")

    if buffer_max is not None and buffer_max <= 0:
        raise ValueError("Buffer invalido.")

    limite = np.inf if buffer_max is None else buffer_max

    x = 0.0
    P = 10000.0

    estimaciones = np.zeros(len(mediciones))
    incertidumbres = np.zeros(len(mediciones))

    for k, medicion in enumerate(mediciones):

        if k > 0:
            # Modelo no lineal de la cola
            propuesta = x + dt * (llegadas[k - 1] - mu)

            # Jacobiano local de la funcion de transicion
            F = 1.0 if 0 < propuesta < limite else 0.0

            # Prediccion
            x = np.clip(propuesta, 0, limite)
            P = F * P * F + Q

        # Correccion con la medicion
        K = P / (P + R)

        x = x + K * (medicion - x)
        x = np.clip(x, 0, limite)

        P = (1 - K) * P

        estimaciones[k] = x
        incertidumbres[k] = np.sqrt(P)

    return estimaciones, incertidumbres
