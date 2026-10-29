"""Escenario D: Estimacion en Linea con Filtro de Kalman sobre el Escenario B.

Homogeneizado con los parametros del informe y del dataset CAIDA:
  - Basado en el Escenario B (pico de streaming con buffer B_max = 50,000 pkts).
  - Telemetria ruidosa con R = 2500 (sigma_v = 50 paquetes).
  - Varianza del proceso Q = 100.
  - Comprobacion de convergencia a K_inf = 0.181 y robustez ante valores atipicos (+-1000 pkt).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola, FiltroKalmanCola
from modelos.config import (
    DIR_RESULTADOS,
    DT,
    DURACION_ESTANDAR,
    BUFFER_MAX,
    CAPACIDAD_NOMINAL,
    LAMBDA_VALLE,
    LAMBDA_PICO,
    Q_ESTANDAR,
    R_ESTANDAR,
)


def ejecutar():
    print("=" * 60)
    print("ESCENARIO D: ESTIMACION EN LINEA CON FILTRO DE KALMAN")
    print("=" * 60)

    n = int(DURACION_ESTANDAR / DT)
    tiempo = np.arange(n) * DT

    # 1. Estado de referencia real: Escenario B con buffer finito B_max = 50,000
    llegadas = np.full(n, LAMBDA_VALLE)
    llegadas[(tiempo >= 100) & (tiempo < 400)] = LAMBDA_PICO

    res_euler = modelo_euler(llegadas, CAPACIDAD_NOMINAL, dt=DT, buffer_max=BUFFER_MAX)
    cola_real = res_euler["cola"]  # Longitud n + 1
    t_cola = res_euler["tiempo"]

    # 2. Generacion de mediciones con ruido gaussiano v_k ~ N(0, R=2500)
    rng = np.random.default_rng(42)
    sigma_v = np.sqrt(R_ESTANDAR)
    mediciones = np.maximum(0.0, cola_real + rng.normal(0, sigma_v, len(cola_real)))

    # Inyeccion controlada de valores atipicos (outliers) en t=150 y t=450
    mediciones_con_atipicos = mediciones.copy()
    if len(mediciones) > 450:
        mediciones_con_atipicos[150] += 1000.0
        mediciones_con_atipicos[450] = max(0.0, mediciones_con_atipicos[450] - 1000.0)

    # 3. Estimacion con Filtro de Kalman
    estimaciones, incertidumbre = estimar_cola(
        llegadas, mediciones_con_atipicos, CAPACIDAD_NOMINAL,
        dt=DT, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX
    )

    # Calculo de ganancia teorica estacionaria K_inf segun el informe:
    # K_inf = (Q + sqrt(Q^2 + 4QR)) / (2R + Q + sqrt(Q^2 + 4QR))
    num_k = Q_ESTANDAR + np.sqrt(Q_ESTANDAR**2 + 4 * Q_ESTANDAR * R_ESTANDAR)
    den_k = 2 * R_ESTANDAR + Q_ESTANDAR + np.sqrt(Q_ESTANDAR**2 + 4 * Q_ESTANDAR * R_ESTANDAR)
    k_inf_teorico = num_k / den_k

    # 4. Evaluacion cuantitativa de desempeno
    mae_med = float(np.mean(np.abs(mediciones_con_atipicos - cola_real)))
    mae_kalman = float(np.mean(np.abs(estimaciones - cola_real)))
    rmse_med = float(np.sqrt(np.mean((mediciones_con_atipicos - cola_real)**2)))
    rmse_kalman = float(np.sqrt(np.mean((estimaciones - cola_real)**2)))
    reduccion_error_pct = (1.0 - (rmse_kalman / rmse_med)) * 100.0

    # Cobertura empirica del intervalo de credibilidad al 95% (+-1.96 sigma)
    cobertura_pct = float(100.0 * np.mean(np.abs(estimaciones - cola_real) <= 1.96 * incertidumbre))

    print(f"  Ganancia estacionaria teorica K_inf: {k_inf_teorico:.4f} (aprox 0.181)")
    print(f"  MAE medicion cruda: {mae_med:.2f} pkt | MAE Kalman: {mae_kalman:.2f} pkt")
    print(f"  RMSE medicion cruda: {rmse_med:.2f} pkt | RMSE Kalman: {rmse_kalman:.2f} pkt")
    print(f"  Reduccion de ruido lograda: {reduccion_error_pct:.1f}%")
    print(f"  Cobertura empirica banda 95%: {cobertura_pct:.1f}%")

    # Guardar resumen en CSV
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    resumen = {
        "K_inf_teorico": round(k_inf_teorico, 4),
        "MAE_Mediciones": round(mae_med, 2),
        "MAE_Kalman": round(mae_kalman, 2),
        "RMSE_Mediciones": round(rmse_med, 2),
        "RMSE_Kalman": round(rmse_kalman, 2),
        "Reduccion_Ruido_pct": round(reduccion_error_pct, 2),
        "Cobertura_95_pct": round(cobertura_pct, 2),
    }
    csv_path = DIR_RESULTADOS / "metricas_escenario_d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen.keys()))
        writer.writeheader()
        writer.writerow(resumen)
    print(f"  Metricas guardadas en: {csv_path}")

    # 5. Grafica de visualizacion limpia
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(t_cola[::3], mediciones_con_atipicos[::3], color="gray", alpha=0.35, s=12, label="Mediciones con ruido y atípicos ($y_k$)")
    ax.plot(t_cola, cola_real, color="black", lw=1.8, label="Estado real - Euler ($q_k$)")
    ax.plot(t_cola, estimaciones, color="#1f77b4", lw=2.2, label=r"Estimación Kalman ($\hat{q}_{k|k}$)")
    ax.fill_between(t_cola, np.maximum(0, estimaciones - 1.96 * incertidumbre),
                    np.minimum(BUFFER_MAX, estimaciones + 1.96 * incertidumbre),
                    color="#1f77b4", alpha=0.2, label="Intervalo de credibilidad 95%")
    ax.axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.5, label=r"Límite de buffer ($B_{\max}=50\mathrm{k}$)")

    # Resaltar atipicos
    ax.annotate("Atípico (+1000)", xy=(150, mediciones_con_atipicos[150]),
                xytext=(150, mediciones_con_atipicos[150] + 6000),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
                fontweight="bold", color="purple", fontsize=9)
    ax.annotate("Atípico (-1000)", xy=(450, mediciones_con_atipicos[450]),
                xytext=(450, mediciones_con_atipicos[450] - 6000),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
                fontweight="bold", color="purple", fontsize=9)

    ax.set_xlabel("Tiempo (s)", fontsize=10)
    ax.set_ylabel("Cola (paquetes)", fontsize=10)
    ax.set_title(rf"Escenario D: Estimación en Línea y Filtro de Kalman ($Q={Q_ESTANDAR:.0f}, R={R_ESTANDAR:.0f}, K_\infty={k_inf_teorico:.3f}$)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = DIR_RESULTADOS / "escenario_d.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Grafica guardada en: {png_path}")
    print("=" * 60)
    return resumen


if __name__ == "__main__":
    ejecutar()
