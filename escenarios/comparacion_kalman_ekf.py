"""Comparacion entre Filtro de Kalman Lineal con Proyeccion y Kalman Extendido (EKF).

Homogeneizado con los parametros unificados del proyecto y CAIDA:
  - Escenario B (pico de streaming con buffer B_max = 50,000).
  - Evaluacion comparativa de MAE, RMSE y comportamiento en las discontinuidades
    de saturacion (q = 0 y q = B_max).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from modelos.kalman_extendido import estimar_cola_ekf
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
    print("COMPARACION: KALMAN LINEAL PROYECTADO VS KALMAN EXTENDIDO (EKF)")
    print("=" * 60)

    n = int(DURACION_ESTANDAR / DT)
    tiempo = np.arange(n) * DT

    # Estado de referencia real: Escenario B con buffer B_max = 50,000
    llegadas = np.full(n, LAMBDA_VALLE)
    llegadas[(tiempo >= 100) & (tiempo < 400)] = LAMBDA_PICO

    res_euler = modelo_euler(llegadas, CAPACIDAD_NOMINAL, dt=DT, buffer_max=BUFFER_MAX)
    cola_real = res_euler["cola"]
    t_cola = res_euler["tiempo"]

    # Mediciones identicas para comparacion justa
    rng = np.random.default_rng(42)
    mediciones = np.maximum(0.0, cola_real + rng.normal(0, np.sqrt(R_ESTANDAR), len(cola_real)))

    # Estimacion Kalman Lineal con proyeccion
    est_kf, sigma_kf = estimar_cola(
        llegadas, mediciones, CAPACIDAD_NOMINAL,
        dt=DT, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX
    )

    # Estimacion Kalman Extendido (EKF por tramos)
    est_ekf, sigma_ekf = estimar_cola_ekf(
        llegadas, mediciones, CAPACIDAD_NOMINAL,
        dt=DT, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX
    )

    def calc_metricas(est):
        err = est - cola_real
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2)))
        }

    met_med = calc_metricas(mediciones)
    met_kf = calc_metricas(est_kf)
    met_ekf = calc_metricas(est_ekf)

    print(f"  Mediciones crudas: MAE = {met_med['MAE']:.2f} pkt | RMSE = {met_med['RMSE']:.2f} pkt")
    print(f"  Kalman Lineal:     MAE = {met_kf['MAE']:.2f} pkt | RMSE = {met_kf['RMSE']:.2f} pkt")
    print(f"  Kalman Extendido:  MAE = {met_ekf['MAE']:.2f} pkt | RMSE = {met_ekf['RMSE']:.2f} pkt")

    # Guardar CSV
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    csv_path = DIR_RESULTADOS / "comparacion_kalman_ekf.csv"
    filas_csv = [
        {"Metodo": "Mediciones", "MAE": round(met_med["MAE"], 2), "RMSE": round(met_med["RMSE"], 2)},
        {"Metodo": "Kalman", "MAE": round(met_kf["MAE"], 2), "RMSE": round(met_kf["RMSE"], 2)},
        {"Metodo": "EKF", "MAE": round(met_ekf["MAE"], 2), "RMSE": round(met_ekf["RMSE"], 2)},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metodo", "MAE", "RMSE"])
        writer.writeheader()
        writer.writerows(filas_csv)
    print(f"  Metricas guardadas en: {csv_path}")

    # Grafica comparativa
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel 1: Trayectorias
    ax1.plot(t_cola, cola_real, color="black", lw=1.8, label="Estado real ($q_k$)")
    ax1.plot(t_cola, est_kf, color="#1f77b4", lw=1.6, label="Kalman Lineal proyectado")
    ax1.plot(t_cola, est_ekf, color="#2ca02c", lw=1.6, linestyle="--", label="Kalman Extendido (EKF)")
    ax1.axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.5, label=r"Límite Buffer ($B_{\max}=50\mathrm{k}$)")
    ax1.set_ylabel("Cola (paquetes)", fontsize=10)
    ax1.set_title("Comparación de Trayectorias: Kalman Lineal vs EKF bajo Saturación", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(alpha=0.3)

    # Panel 2: Error instantaneo absoluto
    err_kf = np.abs(est_kf - cola_real)
    err_ekf = np.abs(est_ekf - cola_real)
    ax2.plot(t_cola, err_kf, color="#1f77b4", lw=1.2, label=f"Error Absoluto Kalman (MAE={met_kf['MAE']:.1f})")
    ax2.plot(t_cola, err_ekf, color="#2ca02c", lw=1.2, linestyle="--", label=f"Error Absoluto EKF (MAE={met_ekf['MAE']:.1f})")
    ax2.set_xlabel("Tiempo (s)", fontsize=10)
    ax2.set_ylabel("Error absoluto (pkt)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = DIR_RESULTADOS / "comparacion_kalman_ekf.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Grafica guardada en: {png_path}")
    print("=" * 60)
    return {"Mediciones": met_med, "Kalman": met_kf, "EKF": met_ekf}


if __name__ == "__main__":
    ejecutar()
