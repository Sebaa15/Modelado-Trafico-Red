"""Pipeline Integrado de las Tres Capas: Bayes -> Euler -> Kalman -> Decisiones.

Implementacion completa y ejecutable segun la Seccion 4 y Seccion 5 del informe
(Elemento3/CasoDeEstudio/modelado-e2.md).
Integra:
  1. Red Bayesiana de contexto (Capa 2) para parametrizar las llegadas y evaluar riesgo basal.
  2. Simulador determinista de Euler con buffer finito B_max = 50,000 (Capa 1).
  3. Filtro de Kalman en linea con bandas de credibilidad al 95% (Capa 3).
  4. Motor de alerta temprana proyectiva a m=15 segundos y recomendacion de capacidad.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from modelos.red_bayesiana import RedBayesianaTrafico
from modelos.kalman import FiltroKalmanCola
from euler.euler import modelo_euler


def simular_euler_informe(N, dt, mu, perfil_llegadas, B_max=50000.0):
    """
    Simulador de colas con Euler explicito, no negatividad y tail-drop
    identico al bloque de la Seccion 4 del informe.
    """
    q = np.zeros(N)
    u = np.zeros(N)
    perdidas = np.zeros(N)

    for k in range(1, N):
        lam = perfil_llegadas(k, dt)
        q_sin_limite = max(0.0, q[k - 1] + dt * (lam - mu))
        q[k] = min(B_max, q_sin_limite)
        perdidas[k] = max(0.0, q_sin_limite - B_max)
        u[k] = lam / mu

    return q, u, perdidas


def ejecutar():
    print("=" * 65)
    print("PIPELINE INTEGRADO: BAYES -> EULER -> KALMAN -> DECISIONES")
    print("=" * 65)

    np.random.seed(42)
    dt = 1.0
    T = 600
    N = int(T / dt)
    B_max = 50000.0

    # -------------------------------------------------------------
    # 1. Capa 2: Inferencia Bayesiana de Contexto
    # -------------------------------------------------------------
    rb = RedBayesianaTrafico()
    lam_esp, prob_sat, dist_lam = rb.inferir_contexto("Pico", "Streaming", "Moderado")

    print("\n=== CAPA 2: INFERENCIA BAYESIANA ===")
    print("Contexto: Lunes 20:00 (Pico), App=Streaming, Historial=Moderado")
    print(f"Distribucion P(Lambda | Contexto): Baja={dist_lam[0]:.2f}, Media={dist_lam[1]:.2f}, Alta={dist_lam[2]:.2f}")
    print(f"Tasa esperada E[lambda]: {lam_esp:.1f} pkt/s | Prob. de Saturacion: {prob_sat*100:.2f}%\n")

    # -------------------------------------------------------------
    # 2. Capa 1: Dinamica Determinista (Euler) - Escenarios A, B, C
    # -------------------------------------------------------------
    perfil_moderado = lambda k, dt: 800.0
    perfil_pico = lambda k, dt: 1500.0 if 100 <= k * dt < 400 else 800.0

    # Escenario A: mu=1000, trafico moderado (800 pkt/s)
    q_A, u_A, p_A = simular_euler_informe(N, dt, mu=1000.0, perfil_llegadas=perfil_moderado, B_max=B_max)
    # Escenario B: mu=1000, hora pico con streaming (1500 pkt/s entre t=100s y t=400s)
    q_B, u_B, p_B = simular_euler_informe(N, dt, mu=1000.0, perfil_llegadas=perfil_pico, B_max=B_max)
    # Escenario C: mu=600 (enlace degradado), trafico moderado (800 pkt/s)
    q_C, u_C, p_C = simular_euler_informe(N, dt, mu=600.0, perfil_llegadas=perfil_moderado, B_max=B_max)

    print("=== CAPA 1: METRICAS DE SIMULACION (EULER) ===")
    print(f"Escenario A -> Util. max: {np.max(u_A):.2f} | Cola max: {np.max(q_A):.0f} pkt | Perdidas: {np.sum(p_A):.0f} pkt")
    print(f"Escenario B -> Util. max: {np.max(u_B):.2f} | Cola max: {np.max(q_B):.0f} pkt | Perdidas: {np.sum(p_B):.0f} pkt")
    print(f"Escenario C -> Util. max: {np.max(u_C):.2f} | Cola max: {np.max(q_C):.0f} pkt | Perdidas: {np.sum(p_C):.0f} pkt\n")

    # -------------------------------------------------------------
    # 3. Capa 3: Filtro de Kalman en Linea y Toma de Decisiones
    # -------------------------------------------------------------
    sigma_proceso = 150.0    # Incertidumbre del modelo (rafagas estocasticas)
    sigma_medicion = 2500.0  # Ruido del sensor de telemetria / SNMP
    Q = sigma_proceso ** 2
    R = sigma_medicion ** 2

    mu_B = 1000.0
    q_real = np.zeros(N)
    y_med = np.zeros(N)
    q_est = np.zeros(N)
    std_est = np.zeros(N)
    alertas_tempranas = []

    kf = FiltroKalmanCola(dt=dt, Q=Q, R=R, x0=0.0, P0=1000.0, B_max=B_max)

    for k in range(1, N):
        lam_k = perfil_pico(k, dt)
        ruido_w = np.random.normal(0, sigma_proceso)
        q_real[k] = np.clip(q_real[k - 1] + dt * (lam_k - mu_B) + ruido_w, 0.0, B_max)

        # Medicion ruidosa y_k = q_k + v_k
        y_med[k] = max(0.0, q_real[k] + np.random.normal(0, sigma_medicion))

        # Paso de estimacion con Kalman
        u_ctrl = lam_k - mu_B
        x_hat, P_k, K_k = kf.paso(u_ctrl, y_med[k])
        q_est[k] = x_hat
        std_est[k] = np.sqrt(P_k)

        # Regla de decision: pronosticar a m=15 segundos
        # Alertar si el limite superior de credibilidad al 95% supera el 80% del buffer
        q_fut, P_fut = kf.predecir_horizonte(u_ctrl, m_pasos=15)
        if q_fut + 1.96 * np.sqrt(P_fut) >= 0.80 * B_max:
            alertas_tempranas.append(k * dt)

    rmse_medicion = float(np.sqrt(np.mean((y_med - q_real) ** 2)))
    rmse_kalman = float(np.sqrt(np.mean((q_est - q_real) ** 2)))
    primera_alerta = float(alertas_tempranas[0]) if alertas_tempranas else None

    # Instante en que la cola real alcanza la saturacion (B_max)
    idx_sat_real = np.where(q_real >= B_max - 1e-3)[0]
    t_sat_real = float(idx_sat_real[0] * dt) if len(idx_sat_real) > 0 else None

    print("=== CAPA 3 Y DECISION: FILTRO DE KALMAN ===")
    print(f"RMSE Medicion Ruidosa: {rmse_medicion:.1f} pkt | RMSE Filtro de Kalman: {rmse_kalman:.1f} pkt")
    reduccion_ruido = (1.0 - (rmse_kalman / rmse_medicion)) * 100.0
    print(f"Reduccion de error / ruido lograda: {reduccion_ruido:.1f}%")
    if primera_alerta is not None:
        anticipacion = (t_sat_real - primera_alerta) if t_sat_real else 0.0
        print(f"Primera alerta temprana de saturacion emitida en t = {primera_alerta:.0f} s")
        print(f"Saturacion real del buffer en t = {t_sat_real:.0f} s (Ventana preventiva de {anticipacion:.0f} s)\n")

    # -------------------------------------------------------------
    # 4. Exportar metricas cuantitativas a CSV
    # -------------------------------------------------------------
    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    resumen_filas = [
        {"Capa": "2_Bayes", "Metrica": "Prob_Saturacion_Contextual", "Valor": round(prob_sat, 4)},
        {"Capa": "2_Bayes", "Metrica": "Lambda_Esperada_pps", "Valor": round(lam_esp, 2)},
        {"Capa": "1_Euler", "Metrica": "Esc_A_ColaMax_pkt", "Valor": round(np.max(q_A), 1)},
        {"Capa": "1_Euler", "Metrica": "Esc_B_ColaMax_pkt", "Valor": round(np.max(q_B), 1)},
        {"Capa": "1_Euler", "Metrica": "Esc_B_Perdidas_pkt", "Valor": round(np.sum(p_B), 1)},
        {"Capa": "1_Euler", "Metrica": "Esc_C_ColaMax_pkt", "Valor": round(np.max(q_C), 1)},
        {"Capa": "1_Euler", "Metrica": "Esc_C_Perdidas_pkt", "Valor": round(np.sum(p_C), 1)},
        {"Capa": "3_Kalman", "Metrica": "RMSE_Medicion_pkt", "Valor": round(rmse_medicion, 2)},
        {"Capa": "3_Kalman", "Metrica": "RMSE_Kalman_pkt", "Valor": round(rmse_kalman, 2)},
        {"Capa": "3_Kalman", "Metrica": "Reduccion_Ruido_pct", "Valor": round(reduccion_ruido, 2)},
        {"Capa": "Decision", "Metrica": "Primera_Alerta_s", "Valor": primera_alerta},
        {"Capa": "Decision", "Metrica": "Saturacion_Real_s", "Valor": t_sat_real},
    ]

    ruta_csv = carpeta / "pipeline_integrado.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Capa", "Metrica", "Valor"])
        writer.writeheader()
        writer.writerows(resumen_filas)
    print(f"Resumen exportado a: {ruta_csv}")

    # -------------------------------------------------------------
    # 5. Graficacion de Resultados (Seccion 4 del informe)
    # -------------------------------------------------------------
    t = np.arange(N) * dt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    # Grafica 1: Comparativa de Escenarios A, B y C (Euler)
    ax[0].plot(t, q_A, label=r"Escenario A (Moderado, $\mu=1000$)", lw=2, color="#1f77b4")
    ax[0].plot(t, q_B, label=r"Escenario B (Pico Streaming, $\mu=1000$)", lw=2, color="#d62728")
    ax[0].plot(t, q_C, label=r"Escenario C (Enlace Degradado, $\mu=600$)", lw=2, linestyle="--", color="#ff7f0e")
    ax[0].axhline(B_max, color="red", linestyle=":", lw=1.8, label=r"Limite Buffer ($B_{\max}=50\mathrm{k}$)")
    ax[0].set_ylabel(r"Cola $q_k$ (paquetes)", fontsize=10)
    ax[0].set_title("Capa 1: Dinámica de Colas Determinista (Euler Explícito)", fontsize=11, fontweight="bold")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(alpha=0.3)

    # Grafica 2: Filtro de Kalman en linea sobre Escenario B
    ax[1].scatter(t[::3], y_med[::3], color="gray", alpha=0.35, s=12, label=r"Mediciones ruidosas ($y_k$)")
    ax[1].plot(t, q_real, color="black", lw=1.5, label=r"Estado verdadero ($q_k$)")
    ax[1].plot(t, q_est, color="blue", lw=2, label=r"Estimación Kalman ($\hat{q}_{k|k}$)")
    ax[1].fill_between(t, np.maximum(0, q_est - 1.96 * std_est), np.minimum(B_max, q_est + 1.96 * std_est),
                       color="blue", alpha=0.2, label="Intervalo de credibilidad 95%")
    if primera_alerta is not None:
        ax[1].axvline(primera_alerta, color="crimson", linestyle="--", lw=1.8,
                      label=f"Alerta temprana ($t={primera_alerta:.0f}$ s)")
    if t_sat_real is not None:
        ax[1].axvline(t_sat_real, color="black", linestyle=":", lw=1.5,
                      label=f"Saturación real ($t={t_sat_real:.0f}$ s)")
    ax[1].axhline(0.80 * B_max, color="darkorange", linestyle=":", lw=1.5, label=r"Umbral Alerta ($0.8 B_{\max}$)")

    ax[1].set_xlabel("Tiempo (s)", fontsize=10)
    ax[1].set_ylabel("Cola (paquetes)", fontsize=10)
    ax[1].set_title("Capa 3: Estimación de Estado en Línea (Kalman) y Alerta Predictiva", fontsize=11, fontweight="bold")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    ruta_png = carpeta / "pipeline_integrado.png"
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {ruta_png}")
    print("=" * 65)

    return {
        "rmse_medicion": rmse_medicion,
        "rmse_kalman": rmse_kalman,
        "primera_alerta": primera_alerta,
        "t_sat_real": t_sat_real
    }


if __name__ == "__main__":
    ejecutar()
