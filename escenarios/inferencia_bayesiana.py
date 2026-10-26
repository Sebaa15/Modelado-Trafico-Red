"""Experimento y evaluacion de la Capa 2: Red Bayesiana de Contexto.

Resuelve la consulta cruzada de Fase 3 y analiza la sensibilidad contextual
frente a variaciones de horario, tipo de aplicacion e historial de congestion.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from modelos.red_bayesiana import RedBayesianaTrafico, inferir_pgmpy, PGMPY_DISPONIBLE


def ejecutar():
    print("=" * 60)
    print("CAPA 2: INFERENCIA CONTEXTUAL (RED BAYESIANA)")
    print("=" * 60)

    rb = RedBayesianaTrafico()

    # 1. Consulta obligatoria de la Fase 3 del informe
    res_fase3 = rb.consulta_fase3()
    print(f"\n[Consulta Fase 3]")
    print(f"Contexto: Lunes 20:00 (Hora=Pico, Dia=Laboral), App=Streaming, Historial=Moderado")
    print(f"  P(Lambda=Baja):  {res_fase3['p_baja']:.4f}")
    print(f"  P(Lambda=Media): {res_fase3['p_media']:.4f}")
    print(f"  P(Lambda=Alta):  {res_fase3['p_alta']:.4f}")
    print(f"  Tasa esperada E[lambda]: {res_fase3['lambda_esperada_pps']:.1f} pkt/s")
    print(f"  Probabilidad de saturacion P(S=1): {res_fase3['prob_saturacion']:.4f} ({res_fase3['prob_saturacion_pct']:.2f}%)")

    # Inferencia con pgmpy si esta disponible
    if PGMPY_DISPONIBLE:
        try:
            res_pgm = inferir_pgmpy({"H": "Pico", "A": "Streaming", "HC": "Moderado"})
            print(f"  [Verificacion con pgmpy] P(S=Si | e): {res_pgm['prob_saturacion']:.4f}")
        except Exception as err:
            print(f"  [pgmpy omitido]: {err}")

    # 2. Barrido contextual sistematico
    combinaciones = []
    horas = ["Valle", "Pico"]
    apps = ["Web", "Streaming"]
    historiales = ["Bajo", "Moderado", "Alto"]

    for h in horas:
        for a in apps:
            for hc in historiales:
                lam_esp, p_sat, dist = rb.inferir_contexto(h, a, hc)
                combinaciones.append({
                    "Hora": h,
                    "Aplicacion": a,
                    "Historial": hc,
                    "P_Baja": round(float(dist[0]), 3),
                    "P_Media": round(float(dist[1]), 3),
                    "P_Alta": round(float(dist[2]), 3),
                    "Lambda_Esperada_pps": round(lam_esp, 1),
                    "Prob_Saturacion_pct": round(p_sat * 100.0, 2),
                })

    # Guardar resultados en CSV
    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    ruta_csv = carpeta / "inferencia_bayesiana.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=combinaciones[0].keys())
        writer.writeheader()
        writer.writerows(combinaciones)
    print(f"\nResultados exportados a: {ruta_csv}")

    # 3. Graficacion comparativa del riesgo de saturacion
    etiquetas = [f"{c['Hora']}\n{c['Aplicacion']}\nHC:{c['Historial'][:3]}" for c in combinaciones]
    riesgos = [c["Prob_Saturacion_pct"] for c in combinaciones]
    tasas = [c["Lambda_Esperada_pps"] for c in combinaciones]

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(etiquetas))
    ancho = 0.45

    colores = ["#2b5c8f" if "Valle" in c["Hora"] else "#d95f02" for c in combinaciones]
    barras = ax1.bar(x, riesgos, ancho, color=colores, alpha=0.85, label="Riesgo Saturacion (%)")
    ax1.set_ylabel("Probabilidad de Saturacion (%)", color="#2b5c8f", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Linea de tasa esperada
    ax2 = ax1.twinx()
    ax2.plot(x, tasas, color="#2ca02c", marker="o", lw=2, label="E[lambda] (pkt/s)")
    ax2.axhline(1000.0, color="gray", linestyle=":", label="Capacidad nominal (mu=1000)")
    ax2.set_ylabel("Tasa esperada E[lambda] (pkt/s)", color="#2ca02c", fontsize=11)
    ax2.set_ylim(700, 1600)

    # Resaltar la consulta Fase 3
    idx_fase3 = [i for i, c in enumerate(combinaciones) if c["Hora"] == "Pico" and c["Aplicacion"] == "Streaming" and c["Historial"] == "Moderado"][0]
    ax1.annotate("Consulta Fase 3\n(82.25%)",
                 xy=(idx_fase3, riesgos[idx_fase3]),
                 xytext=(idx_fase3 - 1.5, riesgos[idx_fase3] + 8),
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
                 fontweight="bold", color="crimson")

    plt.title("Capa 2: Inferencia Contextual en Red Bayesiana (Riesgo y Demanda de Tráfico)", fontsize=12)
    fig.tight_layout()
    ruta_png = carpeta / "inferencia_bayesiana.png"
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {ruta_png}")
    return res_fase3


if __name__ == "__main__":
    ejecutar()
