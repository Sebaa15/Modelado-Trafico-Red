"""Modulo de Visualizacion Unificada y Consolidacion Grafica.

Genera 4 dashboards consolidados de alta calidad editorial, eliminando
redundancias, estandarizando paletas de color, escalas temporales y unidades,
e integrando de forma directa las metricas empiricas del dataset CAIDA
(https://www.caida.org/catalog/datasets/trace_stats/).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from euler.euler import modelo_euler
from modelos.kalman import estimar_cola, FiltroKalmanCola
from modelos.kalman_extendido import estimar_cola_ekf
from modelos.red_bayesiana import RedBayesianaTrafico
from datos.cargador_caida import generar_banco_escenarios_homogeneo, cargar_dataset_caida
from modelos.config import (
    DIR_RESULTADOS,
    DT,
    DURACION_ESTANDAR,
    BUFFER_MAX,
    CAPACIDAD_NOMINAL,
    CAPACIDAD_DEGRADADA,
    Q_ESTANDAR,
    R_ESTANDAR
)


def generar_figura_1_escenarios_euler(banco, dir_out=DIR_RESULTADOS):
    """Figura 1: Dinamica Determinista de Colas y Comparativa de Escenarios Euler."""
    tiempo = banco["tiempo"]
    res_A = modelo_euler(banco["A"]["llegadas"], banco["A"]["capacidad"], dt=DT, buffer_max=BUFFER_MAX)
    res_B = modelo_euler(banco["B"]["llegadas"], banco["B"]["capacidad"], dt=DT, buffer_max=BUFFER_MAX)
    res_C = modelo_euler(banco["C"]["llegadas"], banco["C"]["capacidad"], dt=DT, buffer_max=BUFFER_MAX)
    res_E = modelo_euler(banco["E"]["llegadas"], banco["E"]["capacidad"], dt=DT, buffer_max=BUFFER_MAX)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. Llegadas y capacidad
    axes[0].plot(tiempo, banco["A"]["llegadas"], label="Esc. A (Valle)", color="#2b5c8f", lw=1.6)
    axes[0].plot(tiempo, banco["B"]["llegadas"], label="Esc. B (Pico)", color="#d95f02", lw=1.8)
    axes[0].plot(tiempo, banco["E"]["llegadas"], label="Esc. E (CAIDA)", color="#7570b3", lw=1.2, alpha=0.85)
    axes[0].axhline(CAPACIDAD_NOMINAL, color="black", linestyle="--", lw=1.2, label=r"$\mu$ Nominal (1000 pkt/s)")
    axes[0].axhline(CAPACIDAD_DEGRADADA, color="#e7298a", linestyle=":", lw=1.4, label=r"$\mu$ Degradado (600 pkt/s)")
    axes[0].set_ylabel("Llegadas (pkt/s)", fontsize=9.5)
    axes[0].set_title("Figura 1: Banco de Pruebas Euler Homogeneizado con Estadísticas CAIDA", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].grid(alpha=0.3)

    # 2. Utilizacion del enlace
    axes[1].plot(tiempo, res_A["utilizacion"], label="Esc. A", color="#2b5c8f", lw=1.5)
    axes[1].plot(tiempo, res_B["utilizacion"], label="Esc. B", color="#d95f02", lw=1.8)
    axes[1].plot(tiempo, res_C["utilizacion"], label="Esc. C", color="#e7298a", lw=1.5, linestyle="--")
    axes[1].plot(tiempo, res_E["utilizacion"], label="Esc. E", color="#7570b3", lw=1.2, alpha=0.85)
    axes[1].axhline(1.0, color="red", linestyle="--", lw=1.2, label="Límite estabilidad ($u=1.0$)")
    axes[1].set_ylabel("Carga ofrecida ($u$)", fontsize=9.5)
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    axes[1].grid(alpha=0.3)

    # 3. Evolucion de la cola
    t_c = res_A["tiempo"]
    axes[2].plot(t_c, res_A["cola"], label="Esc. A (Cola = 0)", color="#2b5c8f", lw=1.5)
    axes[2].plot(t_c, res_B["cola"], label="Esc. B (Satura t = 200s)", color="#d95f02", lw=1.8)
    axes[2].plot(t_c, res_C["cola"], label="Esc. C (Satura t = 250s)", color="#e7298a", lw=1.5, linestyle="--")
    axes[2].plot(t_c, res_E["cola"], label="Esc. E (Ráfagas CAIDA)", color="#7570b3", lw=1.4)
    axes[2].axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.6, label=r"Buffer máx ($B_{\max}=50\mathrm{k}$)")
    axes[2].set_ylabel("Cola $q_k$ (paquetes)", fontsize=9.5)
    axes[2].legend(loc="upper right", fontsize=8, ncol=3)
    axes[2].grid(alpha=0.3)

    # 4. Perdidas acumuladas
    axes[3].plot(tiempo, np.cumsum(res_A["perdidas"]), label="Esc. A (0 pkt)", color="#2b5c8f", lw=1.5)
    axes[3].plot(tiempo, np.cumsum(res_B["perdidas"]), label=f"Esc. B ({np.sum(res_B['perdidas']):.0f} pkt)", color="#d95f02", lw=1.8)
    axes[3].plot(tiempo, np.cumsum(res_C["perdidas"]), label=f"Esc. C ({np.sum(res_C['perdidas']):.0f} pkt)", color="#e7298a", lw=1.5, linestyle="--")
    axes[3].plot(tiempo, np.cumsum(res_E["perdidas"]), label=f"Esc. E ({np.sum(res_E['perdidas']):.0f} pkt)", color="#7570b3", lw=1.4)
    axes[3].set_xlabel("Tiempo (s)", fontsize=10)
    axes[3].set_ylabel("Pérdidas tail-drop (pkt)", fontsize=9.5)
    axes[3].legend(loc="upper left", fontsize=8, ncol=2)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    ruta = dir_out / "fig1_escenarios_euler_caida.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def generar_figura_2_estimacion_kalman_ekf(banco, dir_out=DIR_RESULTADOS):
    """Figura 2: Estimacion de Estado, Banda de Credibilidad y Comparacion Kalman vs EKF."""
    tiempo = banco["tiempo"]
    dt = banco["dt"]
    llegadas_B = banco["B"]["llegadas"]
    res_euler = modelo_euler(llegadas_B, CAPACIDAD_NOMINAL, dt=dt, buffer_max=BUFFER_MAX)
    cola_real = res_euler["cola"]
    t_cola = res_euler["tiempo"]

    rng = np.random.default_rng(42)
    mediciones = np.maximum(0.0, cola_real + rng.normal(0, np.sqrt(R_ESTANDAR), len(cola_real)))
    mediciones_atipicas = mediciones.copy()
    mediciones_atipicas[150] += 1000.0
    mediciones_atipicas[450] = max(0.0, mediciones_atipicas[450] - 1000.0)

    est_kf, sigma_kf = estimar_cola(llegadas_B, mediciones_atipicas, CAPACIDAD_NOMINAL, dt=dt, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX)
    est_ekf, sigma_ekf = estimar_cola_ekf(llegadas_B, mediciones_atipicas, CAPACIDAD_NOMINAL, dt=dt, Q=Q_ESTANDAR, R=R_ESTANDAR, buffer_max=BUFFER_MAX)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)

    # 1. Estimacion en linea con bandas al 95%
    axes[0].scatter(t_cola[::3], mediciones_atipicas[::3], color="gray", alpha=0.3, s=10, label="Telemetría con ruido ($y_k$)")
    axes[0].plot(t_cola, cola_real, color="black", lw=1.8, label="Cola real ($q_k$)")
    axes[0].plot(t_cola, est_kf, color="#1f77b4", lw=2.0, label=r"Kalman Lineal ($\hat{q}_{k|k}$)")
    axes[0].fill_between(t_cola, np.maximum(0, est_kf - 1.96 * sigma_kf), np.minimum(BUFFER_MAX, est_kf + 1.96 * sigma_kf),
                         color="#1f77b4", alpha=0.2, label="Banda de credibilidad 95%")
    axes[0].axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.4, label=r"Buffer máx ($B_{\max}=50\mathrm{k}$)")
    axes[0].annotate("Atípico (+1000)", xy=(150, mediciones_atipicas[150]), xytext=(150, mediciones_atipicas[150] + 5000),
                     arrowprops=dict(arrowstyle="->", color="purple"), fontsize=8, color="purple", fontweight="bold")
    axes[0].set_ylabel("Cola (paquetes)", fontsize=9.5)
    axes[0].set_title("Figura 2: Estimación de Estado en Línea, Bandas de Incertidumbre y Comparativa EKF", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Kalman Lineal vs EKF
    axes[1].plot(t_cola, cola_real, color="black", lw=1.4, alpha=0.7, label="Estado real")
    axes[1].plot(t_cola, est_kf, color="#1f77b4", lw=1.6, label="Kalman Lineal proyectado")
    axes[1].plot(t_cola, est_ekf, color="#2ca02c", lw=1.6, linestyle="--", label="Kalman Extendido (EKF)")
    axes[1].set_ylabel("Cola (paquetes)", fontsize=9.5)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. Error absoluto instantaneo
    err_med = np.abs(mediciones_atipicas - cola_real)
    err_kf = np.abs(est_kf - cola_real)
    err_ekf = np.abs(est_ekf - cola_real)
    axes[2].plot(t_cola, err_med, color="gray", alpha=0.35, lw=1.0, label=f"Error Medición cruda (MAE={np.mean(err_med):.1f})")
    axes[2].plot(t_cola, err_kf, color="#1f77b4", lw=1.5, label=f"Error Kalman Lineal (MAE={np.mean(err_kf):.1f})")
    axes[2].plot(t_cola, err_ekf, color="#2ca02c", lw=1.5, linestyle="--", label=f"Error EKF (MAE={np.mean(err_ekf):.1f})")
    axes[2].set_xlabel("Tiempo (s)", fontsize=10)
    axes[2].set_ylabel("Error absoluto (pkt)", fontsize=9.5)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ruta = dir_out / "fig2_estimacion_kalman_ekf.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def generar_figura_3_calibracion_y_diagnostico(banco, dir_out=DIR_RESULTADOS):
    """Figura 3: Calibracion Parametrica (Q, R), Convergencia de Covarianza y Diagnostico NIS."""
    tiempo = banco["tiempo"]
    dt = banco["dt"]
    llegadas = banco["B"]["llegadas"]
    res_euler = modelo_euler(llegadas, CAPACIDAD_NOMINAL, dt=dt, buffer_max=BUFFER_MAX)
    cola_real = res_euler["cola"]

    rng = np.random.default_rng(42)
    mediciones = np.maximum(0.0, cola_real + rng.normal(0, np.sqrt(R_ESTANDAR), len(cola_real)))

    valores_q = [10.0, 100.0, 1000.0]
    valores_r = [625.0, 2500.0, 10000.0]
    rmse_grid = np.zeros((len(valores_q), len(valores_r)))

    for i, q in enumerate(valores_q):
        for j, r in enumerate(valores_r):
            est, _ = estimar_cola(llegadas, mediciones, CAPACIDAD_NOMINAL, dt=dt, Q=q, R=r, buffer_max=BUFFER_MAX)
            rmse_grid[i, j] = np.sqrt(np.mean((est - cola_real)**2))

    # Convergencia de P y K a lo largo del tiempo
    kf = FiltroKalmanCola(dt=dt, Q=Q_ESTANDAR, R=R_ESTANDAR, x0=0.0, P0=1000.0, B_max=BUFFER_MAX)
    p_hist = []
    k_hist = []
    innovaciones = []
    s_hist = []

    for k in range(1, len(cola_real)):
        u = llegadas[k - 1] - CAPACIDAD_NOMINAL
        x_pred = np.clip(kf.x_hat + dt * u, 0.0, BUFFER_MAX)
        p_pred = kf.P + Q_ESTANDAR
        inn = mediciones[k] - x_pred
        s_k = p_pred + R_ESTANDAR
        k_k = p_pred / s_k
        kf.x_hat = np.clip(x_pred + k_k * inn, 0.0, BUFFER_MAX)
        kf.P = (1.0 - k_k) * p_pred

        p_hist.append(kf.P)
        k_hist.append(k_k)
        innovaciones.append(inn)
        s_hist.append(s_k)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Mapa de calor sensibilidad Q/R
    im = axes[0].imshow(rmse_grid, cmap="Blues_r", aspect="auto")
    axes[0].set_xticks(range(len(valores_r)))
    axes[0].set_xticklabels([f"{int(r)}" for r in valores_r])
    axes[0].set_yticks(range(len(valores_q)))
    axes[0].set_yticklabels([f"{int(q)}" for q in valores_q])
    axes[0].set_xlabel(r"Varianza de medición $R$ ($\sigma_v^2$)")
    axes[0].set_ylabel(r"Varianza de proceso $Q$")
    axes[0].set_title(r"Sensibilidad Paramétrica (RMSE)", fontsize=10, fontweight="bold")
    for i in range(len(valores_q)):
        for j in range(len(valores_r)):
            axes[0].text(j, i, f"{rmse_grid[i, j]:.1f}", ha="center", va="center", color="black" if rmse_grid[i,j]>15 else "white", fontweight="bold")

    # Panel 2: Convergencia de Ganancia K_k
    t_ev = np.arange(1, len(cola_real)) * dt
    axes[1].plot(t_ev[:50], k_hist[:50], color="#2b5c8f", lw=2.0, label=r"Ganancia $K_k$")
    axes[1].axhline(0.181, color="red", linestyle="--", lw=1.5, label=r"$K_\infty \approx 0.181$ (Teórica)")
    axes[1].set_xlabel("Tiempo inicial (s)")
    axes[1].set_ylabel("Ganancia de Kalman $K_k$")
    axes[1].set_title(r"Convergencia Rápida a Estado Estacionario ($< 5$ s)", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(alpha=0.3)

    # Panel 3: Diagnostico de Innovaciones y Normalidad
    inn_norm = np.array(innovaciones) / np.sqrt(s_hist)
    axes[2].plot(t_ev, inn_norm, color="#7570b3", lw=0.9, alpha=0.75, label="Innovación normalizada")
    axes[2].axhline(1.96, color="red", linestyle=":", label="Banda 95% ($\pm 1.96$)")
    axes[2].axhline(-1.96, color="red", linestyle=":")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("Residuo normalizado")
    axes[2].set_title("Test de Innovación (Ruido Blanco)", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ruta = dir_out / "fig3_calibracion_y_diagnostico.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def generar_figura_4_pipeline_bayes_caida_decisiones(banco, dir_out=DIR_RESULTADOS):
    """Figura 4: Integracion Contextual Bayes, Traza Real CAIDA y Toma de Decisiones."""
    # 1. Evaluacion Bayesiana
    rb = RedBayesianaTrafico()
    contextos = [
        ("Valle", "Web", "Bajo"),
        ("Valle", "Streaming", "Moderado"),
        ("Pico", "Web", "Moderado"),
        ("Pico", "Streaming", "Moderado")  # Consulta Fase 3
    ]
    riesgos_sat = []
    tasas_esp = []
    labels_ctx = ["Valle / Web", "Valle / Video", "Pico / Web", "Pico / Video\n(Fase 3: 82.2%)"]

    for h, a, hc in contextos:
        lam_e, p_sat, _ = rb.inferir_contexto(h, a, hc)
        riesgos_sat.append(p_sat * 100.0)
        tasas_esp.append(lam_e)

    # 2. Traza CAIDA real con flujos y paquetes
    tiempo = banco["tiempo"]
    dt = banco["dt"]
    llegadas_caida = banco["E"]["llegadas"]
    flujos_caida = banco["E"]["flujos"]

    res_euler_caida = modelo_euler(llegadas_caida, banco["E"]["capacidad"], dt=dt, buffer_max=BUFFER_MAX)
    cola_caida = res_euler_caida["cola"]
    t_c = res_euler_caida["tiempo"]

    # Kalman y alerta temprana predictiva
    rng = np.random.default_rng(42)
    med_caida = np.maximum(0.0, cola_caida + rng.normal(0, np.sqrt(R_ESTANDAR), len(cola_caida)))
    kf = FiltroKalmanCola(dt=dt, Q=Q_ESTANDAR, R=R_ESTANDAR, x0=0.0, P0=1000.0, B_max=BUFFER_MAX)
    cola_est = np.zeros(len(cola_caida))
    alerta_s = None

    for k in range(1, len(cola_caida)):
        u = flujos_caida[k - 1] * 3.8 - banco["E"]["capacidad"]
        x_hat, P_k, _ = kf.paso(llegadas_caida[k - 1] - banco["E"]["capacidad"], med_caida[k])
        cola_est[k] = x_hat
        q_fut, P_fut = kf.predecir_horizonte(u, m_pasos=15)
        if (q_fut + 1.96 * np.sqrt(P_fut)) >= 0.80 * BUFFER_MAX and alerta_s is None:
            alerta_s = tiempo[k - 1]

    idx_sat = np.where(cola_caida >= BUFFER_MAX - 1e-3)[0]
    sat_real_s = float(t_c[idx_sat[0]]) if len(idx_sat) > 0 else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Panel 1: Red Bayesiana
    x_bar = np.arange(len(labels_ctx))
    colores_bar = ["#2b5c8f", "#41b6c4", "#fe9929", "#d95f02"]
    axes[0].bar(x_bar, riesgos_sat, width=0.45, color=colores_bar, alpha=0.85)
    axes[0].set_xticks(x_bar)
    axes[0].set_xticklabels(labels_ctx, fontsize=9)
    axes[0].set_ylabel("Prob. Saturación (%)", fontsize=9.5)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("Figura 4: Pipeline Integral Bayes $\to$ Euler $\to$ Kalman $\to$ Decisiones (Dataset CAIDA)", fontsize=11, fontweight="bold")
    for i, v in enumerate(riesgos_sat):
        axes[0].text(i, v + 2.5, f"{v:.1f}% ({tasas_esp[i]:.0f} pkt/s)", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 2: Traza CAIDA de paquetes y flujos (Leading Indicator)
    axes[1].plot(tiempo, llegadas_caida, color="#1f77b4", lw=1.5, label=r"Paquetes/s ($\lambda_k$) [Traza CAIDA]")
    axes[1].axhline(banco["E"]["capacidad"], color="red", linestyle="--", lw=1.3, label=rf"Capacidad $\mu={banco['E']['capacidad']:.0f}$ pkt/s")
    axes[1].set_ylabel("Paquetes/s", color="#1f77b4", fontsize=9.5)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    ax1_f = axes[1].twinx()
    ax1_f.plot(tiempo, flujos_caida, color="#2ca02c", lw=1.2, alpha=0.8, label=r"Flujos/s ($f_k$) [Leading Indicator]")
    ax1_f.set_ylabel("Flujos/s", color="#2ca02c", fontsize=9.5)
    ax1_f.legend(loc="upper right", fontsize=8)

    # Panel 3: Decisión predictiva y alerta temprana
    axes[2].plot(t_c, cola_caida, color="black", lw=1.6, label="Cola real")
    axes[2].plot(t_c, cola_est, color="#1f77b4", lw=1.8, label=r"Kalman ($\hat{q}_{k|k}$)")
    axes[2].axhline(BUFFER_MAX, color="red", linestyle=":", lw=1.5, label=r"Buffer máx ($B_{\max}=50\mathrm{k}$)")
    axes[2].axhline(0.80 * BUFFER_MAX, color="darkorange", linestyle=":", lw=1.5, label="Umbral preventiva ($0.8 B_{\max}$)")

    if alerta_s is not None:
        axes[2].axvline(alerta_s, color="green", linestyle="--", lw=1.8, label=f"Alerta predictiva flujos ($t={alerta_s:.0f}$s)")
    if sat_real_s is not None:
        axes[2].axvline(sat_real_s, color="crimson", linestyle=":", lw=1.8, label=f"Saturación física real ($t={sat_real_s:.0f}$s)")

    axes[2].set_xlabel("Tiempo (s)", fontsize=10)
    axes[2].set_ylabel("Cola (paquetes)", fontsize=9.5)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ruta = dir_out / "fig4_pipeline_bayes_caida_decisiones.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def generar_todos_los_dashboards():
    """Genera el conjunto completo de 4 figuras unificadas."""
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    banco = generar_banco_escenarios_homogeneo(semilla=42)

    f1 = generar_figura_1_escenarios_euler(banco)
    f2 = generar_figura_2_estimacion_kalman_ekf(banco)
    f3 = generar_figura_3_calibracion_y_diagnostico(banco)
    f4 = generar_figura_4_pipeline_bayes_caida_decisiones(banco)

    print("Dashboards homogeneos generados exitosamente:")
    print(f"  - {f1}")
    print(f"  - {f2}")
    print(f"  - {f3}")
    print(f"  - {f4}")
    return [f1, f2, f3, f4]


if __name__ == "__main__":
    generar_todos_los_dashboards()
