import numpy as np
from filterpy.kalman import KalmanFilter


def estimar_cola(llegadas, mediciones, mu,
                 dt=1.0, Q=100.0, R=2500.0,
                 buffer_max=None):

    llegadas = np.asarray(llegadas, dtype=float)
    mediciones = np.asarray(mediciones, dtype=float)

    if llegadas.ndim != 1 or mediciones.ndim != 1:
        raise ValueError("Las entradas deben ser vectores.")

    if len(llegadas) == 0 or len(mediciones) not in (
        len(llegadas), len(llegadas) + 1
    ):
        raise ValueError("Longitud de series incompatible.")

    if not np.all(np.isfinite(llegadas)):
        raise ValueError("Llegadas invalidas.")

    if not np.all(np.isfinite(mediciones)):
        raise ValueError("Mediciones invalidas.")

    if np.any(llegadas < 0) or mu <= 0 or dt <= 0:
        raise ValueError("Parametros de red invalidos.")

    if Q < 0 or R <= 0:
        raise ValueError("Parametros Q o R invalidos.")

    if buffer_max is not None and buffer_max <= 0:
        raise ValueError("Buffer invalido.")

    limite = np.inf if buffer_max is None else buffer_max

    filtro = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)

    filtro.x = np.array([[0.0]])
    filtro.P = np.array([[10000.0]])

    filtro.F = np.array([[1.0]])
    filtro.B = np.array([[dt]])
    filtro.H = np.array([[1.0]])

    filtro.Q = np.array([[Q]])
    filtro.R = np.array([[R]])

    estimaciones = np.zeros(len(mediciones))
    incertidumbre = np.zeros(len(mediciones))

    for k in range(len(mediciones)):

        # Prediccion del estado
        if k > 0:
            flujo_neto = llegadas[k - 1] - mu

            filtro.predict(
                u=np.array([[flujo_neto]])
            )

            # Restriccion fisica del buffer
            filtro.x[0, 0] = np.clip(
                filtro.x[0, 0], 0, limite
            )

        # Correccion con la medicion
        filtro.update(float(mediciones[k]))

        # Proyeccion a un estado fisicamente valido
        filtro.x[0, 0] = np.clip(
            filtro.x[0, 0], 0, limite
        )

        estimaciones[k] = filtro.x[0, 0]
        incertidumbre[k] = np.sqrt(filtro.P[0, 0])

    return estimaciones, incertidumbre


class FiltroKalmanCola:
    """
    Estimador en espacio de estados 1D (Ecuaciones 5 a 10 del informe):
    x_k = F*x_{k-1} + B*u_k + w_k,  w_k ~ N(0, Q)
    y_k = H*x_k + v_k,              v_k ~ N(0, R)
    """
    def __init__(self, dt=1.0, Q=100.0, R=2500.0, x0=0.0, P0=100.0, B_max=50000.0):
        self.F = 1.0
        self.B = float(dt)
        self.H = 1.0
        self.Q = float(Q)
        self.R = float(R)
        self.x_hat = float(x0)
        self.P = float(P0)
        self.B_max = float(np.inf if B_max is None else B_max)

    def paso(self, u_ctrl, y_medida):
        # 1. Prediccion de estado y covarianza
        x_pred = np.clip(self.F * self.x_hat + self.B * u_ctrl, 0.0, self.B_max)
        P_pred = self.F * self.P * self.F + self.Q

        # 2. Ganancia de Kalman
        K = (P_pred * self.H) / (self.H * P_pred * self.H + self.R)

        # 3. Correccion con la medicion
        self.x_hat = np.clip(x_pred + K * (y_medida - self.H * x_pred), 0.0, self.B_max)
        self.P = (1.0 - K * self.H) * P_pred

        return float(self.x_hat), float(self.P), float(K)

    def predecir_horizonte(self, u_ctrl_esperado, m_pasos=10):
        """Prediccion x_{k+m|k} para anticipar decisiones de red."""
        x_fut = np.clip(self.x_hat + m_pasos * self.B * u_ctrl_esperado, 0.0, self.B_max)
        P_fut = self.P + m_pasos * self.Q
        return float(x_fut), float(P_fut)
