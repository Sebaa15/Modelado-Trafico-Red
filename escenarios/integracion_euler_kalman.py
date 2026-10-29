"""Integracion Euler-Kalman a traves de todo el banco de pruebas homogeneizado.

Aplica el Filtro de Kalman sobre los estados simulados de los Escenarios A, B, C y E (CAIDA),
garantizando coherencia en duracion (600 s), tamano de buffer (50,000 pkts) y ruido (R=2500, Q=100).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from datos.cargador_caida import generar_banco_escenarios_homogeneo
from modelos.config import (
    DIR_RESULTADOS,
    BUFFER_MAX,
    Q_ESTANDAR,
    R_ESTANDAR
)


def ejecutar():
    print("=" * 60)
    print("INTEGRACION EULER-KALMAN SOBRE EL BANCO DE PRUEBAS HOMOGENEO")
    print("=" * 60)

    banco = generar_banco_escenarios_homogeneo(semilla=42)
    tiempo = banco["tiempo"]
    dt = banco["dt"]
    rng = np.random.default_rng(42)

    casos_evaluar = ["A", "B", "C", "E"]
    resumen_filas = []
    resultados_sim = {}

    for clave in casos_evaluar:
        caso = banco[clave]
        llegadas = caso["llegadas"]
        mu = caso["capacidad"]
        buffer_val = caso["buffer_max"]

        # 1. Simulacion determinista con Euler
        res_euler = modelo_euler(llegadas, mu, dt=dt, buffer_max=buffer_val)
        cola_real = res_euler["cola"]

        # 2. Mediciones con ruido de telemetria R=2500 (sigma = 50 pkts)
        mediciones = np.maximum(0.0, cola_real + rng.normal(0, np.sqrt(R_ESTANDAR), len(cola_real)))

        # 3. Estimacion en linea con Kalman
        est_kf, sigma_kf = estimar_cola(
            llegadas, mediciones, mu,
            dt=dt, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=buffer_val
        )

        # 4. Metricas
        err_med = mediciones - cola_real
        err_kf = est_kf - cola_real
        mae_med = float(np.mean(np.abs(err_med)))
        mae_kf = float(np.mean(np.abs(err_kf)))
        rmse_kf = float(np.sqrt(np.mean(err_kf**2)))

        # Cobertura empirica del intervalo de 95%
        cobertura = float(100.0 * np.mean(np.abs(err_kf) <= 1.96 * sigma_kf))
        cola_max = float(np.max(cola_real))
        perd_total = float(np.sum(res_euler["perdidas"]))

        fila = {
            "Escenario": clave,
            "Nombre": caso["nombre"],
            "Capacidad_pps": mu,
            "Buffer_Max_pkt": buffer_val,
            "Cola_Maxima_pkt": round(cola_max, 1),
            "Perdidas_Totales_pkt": round(perd_total, 1),
            "MAE_Mediciones": round(mae_med, 2),
            "MAE_Kalman": round(mae_kf, 2),
            "RMSE_Kalman": round(rmse_kf, 2),
            "Cobertura_95_pct": round(cobertura, 2)
        }
        resumen_filas.append(fila)
        resultados_sim[clave] = {
            "euler": res_euler,
            "mediciones": mediciones,
            "kalman": est_kf,
            "sigma": sigma_kf
        }

        print(f"\n[{caso['nombre']}]")
        print(f"  Cola maxima: {cola_max:.0f} pkt | Perdidas: {perd_total:.0f} pkt")
        print(f"  MAE Medicion: {mae_med:.2f} pkt | MAE Kalman: {mae_kf:.2f} pkt | RMSE Kalman: {rmse_kf:.2f} pkt")
        print(f"  Cobertura banda 95%: {cobertura:.1f}%")

    # Guardar CSV consolidado
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    csv_path = DIR_RESULTADOS / "resumen_euler_kalman.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen_filas[0].keys()))
        writer.writeheader()
        writer.writerows(resumen_filas)
    print(f"\nResumen consolidado exportado a: {csv_path}")

    # Grafica consolidada de 4 paneles (sin repetir ni generar 4 archivos dispersos)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    coords = [((0, 0), "A"), ((0, 1), "B"), ((1, 0), "C"), ((1, 1), "E")]

    for (r, c), clave in coords:
        ax = axes[r, c]
        sim = resultados_sim[clave]
        t_c = sim["euler"]["tiempo"]
        real = sim["euler"]["cola"]
        est = sim["kalman"]
        sig = sim["sigma"]

        ax.scatter(t_c[::4], sim["mediciones"][::4], color="gray", alpha=0.25, s=8, label="Telemetría ($y_k$)")
        ax.plot(t_c, real, color="black", lw=1.6, label="Estado real ($q_k$)")
        ax.plot(t_c, est, color="#1f77b4", lw=1.8, label=r"Kalman ($\hat{q}_{k|k}$)")
        ax.fill_between(t_c, np.maximum(0, est - 1.96 * sig), np.minimum(BUFFER_MAX, est + 1.96 * sig),
                        color="#1f77b4", alpha=0.18)
        ax.axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.2, label=r"$B_{\max}=50\mathrm{k}$")
        ax.set_title(banco[clave]["nombre"], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Cola (paquetes)", fontsize=8.5)
        ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(loc="upper left", fontsize=7.5)
        if r == 1:
            ax.set_xlabel("Tiempo (s)", fontsize=8.5)

    plt.tight_layout()
    png_path = DIR_RESULTADOS / "integracion_euler_kalman.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Grafica comparativa consolidada guardada en: {png_path}")
    print("=" * 60)
    return resumen_filas


if __name__ == "__main__":
    ejecutar()
