import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from modelos.kalman import estimar_cola

dt=1.0
duracion=600
mu=1000.0

tiempo=np.arange(duracion)*dt
llegadas=np.full(duracion,1050.0)

cola_real=np.zeros(duracion)

for k in range(1,duracion):
    cola_real[k]=max(0.0,cola_real[k-1]+dt*(llegadas[k-1]-mu))

generador=np.random.default_rng(42)

R=2500.0
mediciones=cola_real+generador.normal(0,np.sqrt(R),duracion)

mediciones[150]+=1000
mediciones[450]-=1000

estimaciones,incertidumbre=estimar_cola(
    llegadas,mediciones,mu,dt=dt,Q=100.0,R=R
)

mae_mediciones=np.mean(np.abs(mediciones-cola_real))
mae_kalman=np.mean(np.abs(estimaciones-cola_real))
rmse_kalman=np.sqrt(np.mean((estimaciones-cola_real)**2))

print("Escenario D -- Filtro de Kalman")
print(f"MAE mediciones: {mae_mediciones:.2f}")
print(f"MAE Kalman:{mae_kalman:.2f}")
print(f"RMSE Kalman:{rmse_kalman:.2f} ")

Path("resultados").mkdir(exist_ok=True)

plt.figure(figsize=(11,6))
plt.plot(tiempo,cola_real,label="Cola real simulada")
plt.plot(tiempo,mediciones,label="Mediciones con ruido",alpha=0.4)
plt.plot(tiempo, estimaciones,
         label="Estimacion Kalman", linewidth=2)

plt.xlabel("Tiempo (s)")
plt.ylabel("Paquetes en cola")
plt.title("Escenario D - Estimacion con Kalman")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("resultados/escenario_d.png", dpi=150)
plt.show()

# Sensibilidad a Q y R
print("\nPRUEBAS DE CALIBRACION")

for q, r in [(10, 2500), (100, 2500),
             (1000, 2500), (100, 10000)]:

    estimado, _ = estimar_cola(
        llegadas, mediciones, mu,
        dt=dt, Q=q, R=r
    )

    error = np.mean(np.abs(estimado - cola_real))
    print(f"Q={q}, R={r}, MAE={error:.2f}")
