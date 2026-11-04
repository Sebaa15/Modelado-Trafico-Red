import csv 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modelos.kalman import estimar_cola

N=600
dt=1.0
mu=1000.0

llegadas=np.full(N,1050.0)
cola_real=np.zeros(N)

for k in range(1,N):
    cola_real[k]=max(0.0,
                     cola_real[k-1]+dt*(llegadas[k-1]-mu))

rng=np.random.default_rng(42)
mediciones=cola_real+rng.normal(0,50,N)

mediciones_atipicas=mediciones.copy()
mediciones_atipicas[150]+=1000
mediciones_atipicas[450]-=1000

def mae(real,estimado):
    return np.mean(np.abs(real-estimado))
def rmse(real,estimado):
    return np.sqrt(np.mean((real-estimado)**2))
def convergencia_incertidumbre(desviacion):
    referencia=np.median(desviacion[-50:])
    tolerancia=0.05*referencia
    for k in range(len(desviacion)-30):
        ventana=desviacion[k:k+30]
        if np.all(np.abs(ventana-referencia)<=tolerancia):
            return k*dt
    return np.nan

resultados=[]
valores_q=[10,100,1000]
valores_r=[625,2500,10000]

for q in valores_q:
    for r in valores_r:
        estimado,incertidumbre=estimar_cola(
            llegadas,mediciones,mu,dt=dt,Q=q,R=r
        )
        cobertura=np.mean(
            np.abs(estimado-cola_real)<=1.96*incertidumbre)*100
        resultados.append({
            "Q":q,"R":r,"MAE":mae(cola_real,estimado),
            "RMSE":rmse(cola_real,estimado),"Cobertura":cobertura,
            "Convergencia":convergencia_incertidumbre(incertidumbre)
        })

#Ordenar por erro RMSE
resultados.sort(key=lambda x:x["RMSE"])
print("\nCALIBRACION FILTRO")
for fila in resultados:
    print(
        f"Q={fila['Q']:4} R={fila['R']:5} "
        f"MAE={fila['MAE']:.2f} "
        f"RMSE={fila['RMSE']:.2f} "
        f"Cobertura={fila['Cobertura']:.1f}% "
        f"Estabilidad={fila['Convergencia']:.0f}s"
    )
mejor=resultados[0]

print("\nMEJOR COMBINACION EN ESTA SIMULACION")
print(f"Q = {mejor['Q']}")
print(f"R = {mejor['R']}")

q=mejor['Q']
r=mejor['R']

estimacion_normal, _ = estimar_cola(
    llegadas, mediciones, mu,
    dt=dt, Q=q, R=r
)

estimacion_atipica, _ = estimar_cola(
    llegadas, mediciones_atipicas, mu,
    dt=dt, Q=q, R=r
)

print("\nROBUSTEZ ANTE VALORES ATIPICOS")
print(
    "RMSE sin atipicos:",
    round(rmse(cola_real, estimacion_normal), 2)
)
print(
    "RMSE con atipicos:",
    round(rmse(cola_real, estimacion_atipica), 2)
)
print(
    "RMSE mediciones con atipicos:",
    round(rmse(cola_real, mediciones_atipicas), 2)
)

Path("resultados").mkdir(exist_ok=True)

with open(
    "resultados/calibracion_kalman.csv",
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    escritor = csv.DictWriter(
        archivo, fieldnames=resultados[0].keys()
    )
    escritor.writeheader()
    escritor.writerows(resultados)

matriz = np.zeros((len(valores_q), len(valores_r)))

for fila in resultados:
    i = valores_q.index(fila["Q"])
    j = valores_r.index(fila["R"])
    matriz[i, j] = fila["RMSE"]

plt.figure(figsize=(8, 5))
plt.imshow(matriz, cmap="viridis")
plt.colorbar(label="RMSE")

plt.xticks(range(len(valores_r)), valores_r)
plt.yticks(range(len(valores_q)), valores_q)

plt.xlabel("R - Ruido de medicion")
plt.ylabel("Q - Ruido de proceso")
plt.title("Sensibilidad del filtro de Kalman")

for i in range(len(valores_q)):
    for j in range(len(valores_r)):
        plt.text(
            j, i, f"{matriz[i, j]:.1f}",
            ha="center", va="center", color="white"
        )

plt.tight_layout()
plt.savefig("resultados/calibracion_kalman.png", dpi=150)
plt.show()

inicio, fin = 130, 175
tiempo = np.arange(inicio, fin)

plt.figure(figsize=(10, 5))

plt.plot(
    tiempo, cola_real[inicio:fin],
    label="Cola simulada"
)
plt.plot(
    tiempo, mediciones_atipicas[inicio:fin],
    label="Mediciones con atipico", alpha=0.6
)
plt.plot(
    tiempo, estimacion_atipica[inicio:fin],
    label="Estimacion Kalman", linewidth=2
)

plt.xlabel("Tiempo (s)")
plt.ylabel("Paquetes en cola")
plt.title("Respuesta de Kalman ante valor atipico")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados/atipicos_kalman.png", dpi=150)
plt.show()
