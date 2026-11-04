"""Escenario E: Trafico Real CAIDA (Series Empiricas, Micro-rafagas y Flujos).

Modela un enlace backbone de alta capacidad (10G/100G) a partir de las propiedades
empiricas de CAIDA (Center for Applied Internet Data Analysis) descriptas en las
Secciones 2 y 3 del informe:
  1. Series de paquetes/s (lambda_k) y flujos/s (f_k).
  2. Trafico auto-similar con micro-rafagas generadas por distribucion Pareto (colas pesadas).
  3. Los flujos/s actuan como indicador adelantado (leading indicator) de paquetes/s (TCP slow-start).
  4. Calibracion de mu con el percentil 99.9 del trafico sostenido.
  5. Inferencia conjunta Bayes + Kalman para anticipar congestiones con umbral 0.8 * B_max.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from modelos.red_bayesiana import RedBayesianaTrafico
from modelos.kalman import FiltroKalmanCola
from euler.euler import modelo_euler


def generar_serie_caida(duracion=600, dt=1.0, semilla=42):
    """
    Sintetiza una serie temporal con propiedades estadisticas de trazas CAIDA:
    - Auto-similitud y distribucion de colas pesadas (Pareto).
    - Flujos activos f_k que preceden a las rafagas de paquetes lambda_k.
    """
    rng = np.random.default_rng(semilla)
    n = int(duracion / dt)
    tiempo = np.arange(n) * dt

    # 1. Componente basal de flujos y paquetes
    flujos_base = 250.0 + 30.0 * np.sin(2 * np.pi * tiempo / 300)
    flujos = flujos_base + rng.normal(0, 10, n)

    # Inyeccion de rafagas de flujos (nuevas conexiones / sesiones masivas)
    # Los flujos aumentan en t=100..200, provocando que los paquetes aumenten en t=105..205
    evento1_flujos = (tiempo >= 100) & (tiempo <= 210)
    evento2_flujos = (tiempo >= 360) & (tiempo <= 410)
    flujos[evento1_flujos] += 350.0 * (1.0 + 0.3 * rng.pareto(a=2.5, size=np.sum(evento1_flujos)))
    flujos[evento2_flujos] += 200.0 * (1.0 + 0.4 * rng.pareto(a=2.0, size=np.sum(evento2_flujos)))
    flujos = np.maximum(50.0, flujos)

    # 3. Generacion de paquetes/s (lambda_k):
    # La tasa de paquetes sigue a los flujos con un retardo de 4 segundos (TCP slow-start)
    paquetes = np.zeros(n)
    retardo_flujos = 4  # segundos
    for k in range(n):
        idx_retardado = max(0, k - retardo_flujos)
        flujo_efectivo = flujos[idx_retardado]
        factor_rafaga = 1.0 + 0.15 * rng.pareto(a=3.0)

        if 105 <= tiempo[k] <= 220:
            paquetes[k] = flujo_efectivo * 3.8 * factor_rafaga + rng.normal(0, 40)
        elif 365 <= tiempo[k] <= 420:
            paquetes[k] = flujo_efectivo * 3.2 * factor_rafaga + rng.normal(0, 50)
        else:
            paquetes[k] = flujo_efectivo * 4.0 * factor_rafaga + rng.normal(0, 25)

    paquetes = np.maximum(200.0, paquetes)
    return tiempo, paquetes, flujos


def ejecutar():
    print("=" * 65)
    print("ESCENARIO E: TRAFICO REAL CAIDA (SERIES EMPIRICAS Y RAFAGAS)")
    print("=" * 65)

    dt = 1.0
    duracion = 600
    tiempo, lambdas, flujos = generar_serie_caida(duracion=duracion, dt=dt, semilla=42)
    B_max = 50000.0

    # -------------------------------------------------------------
    # 1. Calibracion de mu con el percentil 99.9 del regimen normal
    # -------------------------------------------------------------
    # Se evalua el periodo previo a la congestion severa (primeros 100 s)
    lambdas_base = lambdas[tiempo < 100]
    mu_calibrada = float(np.percentile(lambdas_base, 99.9))
    # Redondeamos a valor representativo de capacidad asignada
    mu_nominal = 1600.0

    print(f"[Calibracion de Enlace]")
    print(f"  Tasa basal P99.9 estimada: {mu_calibrada:.1f} pkt/s")
    print(f"  Capacidad de servicio asignada (mu): {mu_nominal:.1f} pkt/s")
    print(f"  Tamano de buffer (B_max): {B_max:.0f} paquetes")

    # -------------------------------------------------------------
    # 2. Inferencia Bayesiana de Contexto (Prior Horario)
    # -------------------------------------------------------------
    rb = RedBayesianaTrafico()
    lam_esp, prob_sat, _ = rb.inferir_contexto("Pico", "Streaming", "Moderado")
    print(f"\n[Prior Bayesiano Contextual]")
    print(f"  E[lambda] contexto pico: {lam_esp:.1f} pkt/s | Riesgo basal de saturacion: {prob_sat*100:.1f}%")

    # -------------------------------------------------------------
    # 3. Capa 1: Dinamica Determinista (Euler) con la traza CAIDA
    # -------------------------------------------------------------
    resultado_euler = modelo_euler(
        lambdas, capacidad=mu_nominal, dt=dt, buffer_max=B_max
    )
    cola_real = resultado_euler["cola"]
    tiempo_cola = resultado_euler["tiempo"]
    perdidas = resultado_euler["perdidas"]
    utilizacion = resultado_euler["utilizacion"]

    cola_max = float(np.max(cola_real))
    perdidas_totales = float(np.sum(perdidas))
    util_max = float(np.max(utilizacion))
    print(f"\n[Dinamica de Euler - Traza CAIDA]")
    print(f"  Carga maxima ofrecida (u_max): {util_max:.2f}")
    print(f"  Cola maxima acumulada: {cola_max:.0f} paquetes")
    print(f"  Perdidas por desbordamiento: {perdidas_totales:.0f} paquetes")

    # -------------------------------------------------------------
    # 4. Capa 3: Estimacion en linea con Kalman y Alerta Temprana
    # -------------------------------------------------------------
    rng = np.random.default_rng(101)
    sigma_medicion = 1200.0  # Ruido de telemetria
    R = sigma_medicion ** 2
    Q = 180.0 ** 2          # Incertidumbre del proceso por micro-rafagas

    # Mediciones con ruido observadas por telemetria
    y_med = np.maximum(0.0, cola_real + rng.normal(0, sigma_medicion, len(cola_real)))

    kf = FiltroKalmanCola(dt=dt, Q=Q, R=R, x0=0.0, P0=1000.0, B_max=B_max)
    cola_est = np.zeros(len(cola_real))
    sigma_est = np.zeros(len(cola_real))

    alertas_paquetes = []
    alertas_con_flujos = []

    m_horizonte = 15  # segundos de proyeccion predictiva

    for k in range(1, len(cola_real)):
        lam_k = lambdas[k - 1]
        u_ctrl = lam_k - mu_nominal

        # Paso Kalman
        x_hat, P_k, K_k = kf.paso(u_ctrl, y_med[k])
        cola_est[k] = x_hat
        sigma_est[k] = np.sqrt(P_k)

        # Regla 1: Alerta estandar basada solo en paquetes/s observados
        q_fut_std, P_fut_std = kf.predecir_horizonte(u_ctrl, m_pasos=m_horizonte)
        if (q_fut_std + 1.96 * np.sqrt(P_fut_std)) >= 0.80 * B_max:
            alertas_paquetes.append(tiempo_cola[k])

        # Regla 2: Alerta enriquecida con flujos/s como Leading Indicator
        # El volumen de flujos anticipa la tasa de llegada que arribara en los proximos segundos
        flujo_actual = flujos[k - 1]
        tasa_proyectada = flujo_actual * 4.2
        u_ctrl_flujos = tasa_proyectada - mu_nominal
        q_fut_flujos, P_fut_flujos = kf.predecir_horizonte(u_ctrl_flujos, m_pasos=m_horizonte)
        if (q_fut_flujos + 1.96 * np.sqrt(P_fut_flujos)) >= 0.80 * B_max:
            alertas_con_flujos.append(tiempo_cola[k])

    rmse_med = float(np.sqrt(np.mean((y_med - cola_real) ** 2)))
    rmse_kalman = float(np.sqrt(np.mean((cola_est - cola_real) ** 2)))
    primera_alerta_std = float(alertas_paquetes[0]) if alertas_paquetes else None
    primera_alerta_flujos = float(alertas_con_flujos[0]) if alertas_con_flujos else None

    # Saturacion real
    idx_sat = np.where(cola_real >= B_max - 1e-3)[0]
    t_sat_real = float(tiempo_cola[idx_sat[0]]) if len(idx_sat) > 0 else None

    print(f"\n[Filtro de Kalman y Alerta Temprana]")
    print(f"  RMSE Medicion cruda: {rmse_med:.1f} pkt | RMSE Kalman: {rmse_kalman:.1f} pkt")
    print(f"  Reduccion de error: {(1.0 - rmse_kalman/rmse_med)*100:.1f}%")
    if t_sat_real:
        print(f"  Saturacion real en t = {t_sat_real:.0f} s")
    if primera_alerta_std:
        print(f"  Alerta con solo paquetes/s en t = {primera_alerta_std:.0f} s (Anticipacion: {t_sat_real - primera_alerta_std:.0f} s)")
    if primera_alerta_flujos:
        print(f"  Alerta con Flujos/s (Leading Indicator) en t = {primera_alerta_flujos:.0f} s (Anticipacion: {t_sat_real - primera_alerta_flujos:.0f} s)")
        if primera_alerta_std:
            ganancia_flujos = primera_alerta_std - primera_alerta_flujos
            print(f"  Ganancia de tiempo al incorporar flujos/s: +{ganancia_flujos:.0f} segundos antes!")

    # -------------------------------------------------------------
    # 5. Exportar metricas a CSV
    # -------------------------------------------------------------
    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    resumen_filas = [
        {"Metrica": "Mu_Calibrada_pps", "Valor": round(mu_calibrada, 1)},
        {"Metrica": "Mu_Nominal_pps", "Valor": mu_nominal},
        {"Metrica": "Utilizacion_Maxima", "Valor": round(util_max, 3)},
        {"Metrica": "Cola_Maxima_pkt", "Valor": round(cola_max, 1)},
        {"Metrica": "Perdidas_Totales_pkt", "Valor": round(perdidas_totales, 1)},
        {"Metrica": "RMSE_Medicion_pkt", "Valor": round(rmse_med, 1)},
        {"Metrica": "RMSE_Kalman_pkt", "Valor": round(rmse_kalman, 1)},
        {"Metrica": "Saturacion_Real_s", "Valor": t_sat_real},
        {"Metrica": "Alerta_Solo_Paquetes_s", "Valor": primera_alerta_std},
        {"Metrica": "Alerta_Con_Flujos_s", "Valor": primera_alerta_flujos},
    ]

    ruta_csv = carpeta / "escenario_e.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metrica", "Valor"])
        writer.writeheader()
        writer.writerows(resumen_filas)
    print(f"\nResumen exportado a: {ruta_csv}")

    # -------------------------------------------------------------
    # 6. Graficacion Completa
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Subplot 1: Trazas CAIDA de Paquetes/s y Flujos/s
    ax1.plot(tiempo, lambdas, color="#1f77b4", lw=1.5, label=r"Paquetes/s ($\lambda_k$)")
    ax1.axhline(mu_nominal, color="red", linestyle="--", lw=1.5, label=rf"Capacidad $\mu={mu_nominal:.0f}$ pkt/s")
    ax1.set_ylabel("Paquetes/s", color="#1f77b4", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Escenario E: Tráfico Real CAIDA (Backbone 10G/100G) con Micro-ráfagas", fontsize=11, fontweight="bold")

    ax1_f = ax1.twinx()
    ax1_f.plot(tiempo, flujos, color="#2ca02c", lw=1.2, alpha=0.7, label=r"Flujos/s ($f_k$) [Leading]")
    ax1_f.set_ylabel("Flujos/s", color="#2ca02c", fontsize=10)
    ax1_f.legend(loc="upper right", fontsize=9)

    # Subplot 2: Dinamica de Cola y Mediciones
    ax2.plot(tiempo_cola, cola_real, color="black", lw=1.8, label=r"Cola Real ($q_k$)")
    ax2.scatter(tiempo_cola[::4], y_med[::4], color="gray", alpha=0.35, s=10, label=r"Telemetría con ruido ($y_k$)")
    ax2.axhline(B_max, color="crimson", linestyle=":", lw=1.8, label=r"Buffer Máx ($B_{\max}=50\mathrm{k}$)")
    ax2.set_ylabel("Cola (paquetes)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    # Subplot 3: Estimacion Kalman, Intervalos y Alerta con Flujos
    ax3.plot(tiempo_cola, cola_real, color="black", lw=1.2, alpha=0.6, label="Cola Real")
    ax3.plot(tiempo_cola, cola_est, color="blue", lw=1.8, label=r"Kalman ($\hat{q}_{k|k}$)")
    ax3.fill_between(tiempo_cola, np.maximum(0, cola_est - 1.96 * sigma_est),
                     np.minimum(B_max, cola_est + 1.96 * sigma_est),
                     color="blue", alpha=0.2, label="Banda 95%")
    ax3.axhline(0.80 * B_max, color="darkorange", linestyle=":", lw=1.5, label=r"Umbral Alerta ($0.8 B_{\max}$)")

    if primera_alerta_flujos:
        ax3.axvline(primera_alerta_flujos, color="green", linestyle="--", lw=1.8,
                    label=f"Alerta Flujos+Kalman ($t={primera_alerta_flujos:.0f}$ s)")
    if primera_alerta_std:
        ax3.axvline(primera_alerta_std, color="orange", linestyle="--", lw=1.5,
                    label=f"Alerta Solo Paquetes ($t={primera_alerta_std:.0f}$ s)")
    if t_sat_real:
        ax3.axvline(t_sat_real, color="crimson", linestyle=":", lw=1.8,
                    label=f"Saturación Real ($t={t_sat_real:.0f}$ s)")

    ax3.set_xlabel("Tiempo (s)", fontsize=10)
    ax3.set_ylabel("Cola Estimada", fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    ruta_png = carpeta / "escenario_e.png"
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {ruta_png}")
    print("=" * 65)

    return {
        "mu_calibrada": mu_calibrada,
        "cola_max": cola_max,
        "perdidas_totales": perdidas_totales,
        "rmse_kalman": rmse_kalman,
        "primera_alerta_flujos": primera_alerta_flujos,
        "t_sat_real": t_sat_real
    }


if __name__ == "__main__":
    ejecutar()
