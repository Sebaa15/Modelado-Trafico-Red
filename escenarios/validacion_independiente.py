"""Seleccion de Q/R con entrenamiento y evaluacion independientes.

Modelo sintetico con incertidumbre real de proceso (desv. 8 paquetes)
y medicion (desv. 50 paquetes), semillas distintas para ajustar y
validar. Las métricas no representan datos reales de CAIDA.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modelos.kalman import estimar_cola


N = 500
DT = 1.0
MU = 1000.0
Q_VALORES = [16.0, 64.0, 256.0]
R_VALORES = [625.0, 2500.0, 10000.0]
SEMILLAS_AJUSTE = [12, 13, 14]
SEMILLAS_PRUEBA = [112, 113, 114]


def generar_experimento(semilla):
    rng = np.random.default_rng(semilla)
    instantes = np.arange(N)
    llegadas = 1100.0 + 70.0 * np.sin(2 * np.pi * instantes / 150.0)
    q_real = np.zeros(N + 1)
    perturbaciones = rng.normal(0, 8.0, N)
    for k in range(N):
        q_real[k + 1] = max(0.0, q_real[k] + DT *
                            (llegadas[k] - MU) + perturbaciones[k])
    mediciones = q_real + rng.normal(0, 50.0, N + 1)
    return llegadas, q_real, mediciones


def rmse_prueba(q, r, semillas):
    errores = []
    for semilla in semillas:
        llegadas, real, mediciones = generar_experimento(semilla)
        estimado, _ = estimar_cola(llegadas, mediciones, MU,
                                   dt=DT, Q=q, R=r)
        # Omite el tramo inicial para comparar desempeño estable
        errores.append(float(np.sqrt(np.mean((real[20:] - estimado[20:]) ** 2))))
    return float(np.mean(errores))


def ejecutar():
    filas = []
    for q in Q_VALORES:
        for r in R_VALORES:
            filas.append({
                "Q": q, "R": r,
                "RMSE_ajuste": round(rmse_prueba(q, r, SEMILLAS_AJUSTE), 4),
                "RMSE_prueba": round(rmse_prueba(q, r, SEMILLAS_PRUEBA), 4)
            })
    mejor = min(filas, key=lambda f: f["RMSE_ajuste"])
    fila_mejor = next(f for f in filas if f["Q"] == mejor["Q"] and f["R"] == mejor["R"])

    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    with open(carpeta / "validacion_independiente.csv", "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)

    plt.figure(figsize=(8, 5))
    matriz = np.array([[next(f["RMSE_prueba"] for f in filas if f["Q"] == q and f["R"] == r)
                        for r in R_VALORES] for q in Q_VALORES])
    plt.imshow(matriz)
    plt.xticks(range(len(R_VALORES)), [str(int(r)) for r in R_VALORES])
    plt.yticks(range(len(Q_VALORES)), [str(int(q)) for q in Q_VALORES])
    plt.xlabel("R")
    plt.ylabel("Q")
    plt.title("RMSE de prueba independiente por Q y R")
    plt.colorbar(label="RMSE (paquetes)")
    for i in range(len(Q_VALORES)):
        for j in range(len(R_VALORES)):
            plt.text(j, i, f"{matriz[i,j]:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(carpeta / "validacion_independiente.png", dpi=150)
    plt.close()

    print("\nVALIDACION INDEPENDIENTE")
    for f in filas:
        print(f"Q={f['Q']:.0f}, R={f['R']:.0f}, ajuste={f['RMSE_ajuste']:.2f}, prueba={f['RMSE_prueba']:.2f}")
    print(f"Seleccion en ajuste: Q={mejor['Q']:.0f}, R={mejor['R']:.0f}; "
          f"RMSE en prueba independiente: {fila_mejor['RMSE_prueba']:.2f}")
    return filas


if __name__ == "__main__":
    ejecutar()
