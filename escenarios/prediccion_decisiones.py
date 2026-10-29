"""Prediccion anticipada y evaluacion de decisiones de capacidad sobre el Escenario B.

Homogeneizado con los parametros unificados del proyecto y CAIDA:
  - Duracion total 600 s, buffer B_max = 50,000 paquetes.
  - Instante de corte t_corte = 150 s (durante la hora pico, antes de la saturacion real en t=200 s).
  - Horizonte predictivo: 60 s.
  - Compara la capacidad actual (mu=1000 pkt/s) frente a la alternativa (mu=1500 pkt/s).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola
from modelos.prediccion import predecir_cola
from modelos.decisiones import comparar_capacidades, evaluar_congestion
from modelos.config import (
    DIR_RESULTADOS,
    DT,
    DURACION_ESTANDAR,
    BUFFER_MAX,
    CAPACIDAD_NOMINAL,
    LAMBDA_VALLE,
    LAMBDA_PICO,
    Q_ESTANDAR,
    R_ESTANDAR
)


def ejecutar():
    print("=" * 60)
    print("PREDICCION Y TOMA DE DECISIONES DE CAPACIDAD (ESCENARIO B)")
    print("=" * 60)

    n = int(DURACION_ESTANDAR / DT)
    tiempo = np.arange(n) * DT
    instante_corte = 150   # 150 s
    horizonte = 60         # 60 s de horizonte futuro

    # Perfil del Escenario B
    llegadas = np.full(n, LAMBDA_VALLE)
    llegadas[(tiempo >= 100) & (tiempo < 400)] = LAMBDA_PICO

    # Estado de referencia con Euler
    referencia = modelo_euler(llegadas, CAPACIDAD_NOMINAL, dt=DT, buffer_max=BUFFER_MAX)
    cola_real = referencia["cola"]

    # Observaciones ruidosas hasta el instante de corte
    rng = np.random.default_rng(42)
    obs_hasta_corte = cola_real[:instante_corte + 1] + rng.normal(0, np.sqrt(R_ESTANDAR), instante_corte + 1)
    obs_hasta_corte = np.maximum(0.0, obs_hasta_corte)

    # Estimacion con Kalman hasta el corte
    estimaciones, _ = estimar_cola(
        llegadas[:instante_corte], obs_hasta_corte, CAPACIDAD_NOMINAL,
        dt=DT, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX
    )
    estado_estimado_corte = float(estimaciones[-1])

    # Perfil supuesto en el horizonte futuro
    perfil_futuro = llegadas[instante_corte:instante_corte + horizonte]
    nueva_capacidad = float(np.max(perfil_futuro))  # 1500 pkt/s

    # Comparacion contrafactual de capacidades
    comparacion = comparar_capacidades(
        estado_estimado_corte, perfil_futuro, CAPACIDAD_NOMINAL,
        nueva_capacidad, dt=DT, buffer_max=BUFFER_MAX
    )
    pronostico_base = comparacion["actual"]
    pronostico_ampliado = comparacion["alternativa"]
    diag_base = comparacion["diagnostico_actual"]
    diag_ampliado = comparacion["diagnostico_alternativo"]

    # Realidad futura observada segun Euler
    realidad_futura = cola_real[instante_corte:instante_corte + horizonte + 1]
    mae_pred = float(np.mean(np.abs(pronostico_base["cola"] - realidad_futura)))

    print(f"  Instante de corte: {instante_corte} s | Horizonte: {horizonte} s")
    print(f"  Cola estimada en corte: {estado_estimado_corte:.1f} pkt (Real: {cola_real[instante_corte]:.1f} pkt)")
    print(f"  Diagnostico con capacidad actual (mu={CAPACIDAD_NOMINAL:.0f}): {diag_base['nivel']}")
    print(f"    - Tiempo hasta llenar buffer desde corte: {diag_base['tiempo_buffer_lleno_s']} s")
    print(f"    - Perdidas previstas: {diag_base['paquetes_perdidos']:.0f} pkt")
    print(f"  Diagnostico con capacidad ampliada (mu={nueva_capacidad:.0f}): {diag_ampliado['nivel']}")
    print(f"    - Perdidas previstas: {diag_ampliado['paquetes_perdidos']:.0f} pkt")
    print(f"  MAE de prediccion vs realidad: {mae_pred:.2f} pkt")

    # Guardar resumen en CSV
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    resumen = {
        "corte_s": instante_corte,
        "horizonte_s": horizonte,
        "cola_estimada_corte": round(estado_estimado_corte, 2),
        "capacidad_actual_pps": CAPACIDAD_NOMINAL,
        "capacidad_alternativa_pps": nueva_capacidad,
        "alerta_actual": diag_base["nivel"],
        "tiempo_buffer_lleno_desde_corte_s": diag_base["tiempo_buffer_lleno_s"],
        "perdidas_previstas_actual": round(diag_base["paquetes_perdidos"], 1),
        "perdidas_previstas_alternativa": round(diag_ampliado["paquetes_perdidos"], 1),
        "mae_prediccion_vs_euler": round(mae_pred, 2)
    }
    csv_path = DIR_RESULTADOS / "prediccion_decisiones.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen.keys()))
        writer.writeheader()
        writer.writerow(resumen)
    print(f"  Resumen exportado a: {csv_path}")

    # Grafica limpia
    t_abs = instante_corte + pronostico_base["tiempo"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(t_abs, realidad_futura, color="black", lw=2.0, label="Realidad futura - Euler")
    ax.plot(t_abs, pronostico_base["cola"], color="#d62728", lw=1.8, linestyle="--",
            label=f"Pronóstico $\mu={CAPACIDAD_NOMINAL:.0f}$ (Satura en {diag_base['tiempo_buffer_lleno_s']:.0f}s)")
    ax.plot(t_abs, pronostico_ampliado["cola"], color="#2ca02c", lw=1.8,
            label=f"Alternativa ampliada $\mu={nueva_capacidad:.0f}$ (0 pérdidas)")
    ax.axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.5, label=r"Límite de buffer ($B_{\max}=50\mathrm{k}$)")
    ax.axvline(instante_corte, color="gray", linestyle="-.", lw=1.2, label=f"Corte ($t={instante_corte}$s)")

    ax.set_xlabel("Tiempo (s)", fontsize=10)
    ax.set_ylabel("Cola (paquetes)", fontsize=10)
    ax.set_title(f"Predicción y Decisión de Capacidad (Corte $t={instante_corte}$s, Horizonte {horizonte}s)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = DIR_RESULTADOS / "prediccion_decisiones.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Grafica guardada en: {png_path}")
    print("=" * 60)
    return resumen


if __name__ == "__main__":
    ejecutar()
