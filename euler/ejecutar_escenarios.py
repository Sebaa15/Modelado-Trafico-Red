import numpy as np
import matplotlib.pyplot as plt
from euler import modelo_euler


# --------------------------------------------------
# 1. Parametros de simulacion
# --------------------------------------------------

dt = 1.0
duracion = 1200.0
tiempos_intervalos = np.arange(0, duracion, dt)

capacidad_a = 1000.0
capacidad_b = 1000.0
capacidad_c = 600.0


# --------------------------------------------------
# 2. Definir las llegadas de los escenarios
# --------------------------------------------------

llegadas_a = np.full(len(tiempos_intervalos), 800.0)

llegadas_b = np.full(len(tiempos_intervalos), 800.0)
pico = (tiempos_intervalos >= 200) & (tiempos_intervalos < 800)
llegadas_b[pico] = 1500.0

llegadas_c = np.full(len(tiempos_intervalos), 800.0)


# --------------------------------------------------
# 3. Ejecutar Euler sin limite de buffer
# --------------------------------------------------

escenarios = {
    "A": modelo_euler(llegadas_a, capacidad_a, dt),
    "B": modelo_euler(llegadas_b, capacidad_b, dt),
    "C": modelo_euler(llegadas_c, capacidad_c, dt)
}


# --------------------------------------------------
# 4. Calcular e imprimir metricas
# --------------------------------------------------

for nombre, resultado in escenarios.items():

    cola = resultado["cola"]
    utilizacion = resultado["utilizacion"]
    llegadas = resultado["llegadas"]

    indices = np.where(utilizacion >= 1.0)[0]

    print("\nESCENARIO", nombre)
    print("-" * 40)
    print("Capacidad:", resultado["capacidad"], "paquetes/s")
    print("Cola maxima:", np.max(cola), "paquetes")
    print("Cola final:", cola[-1], "paquetes")
    print(
        "Utilizacion media:",
        round(np.mean(utilizacion) * 100, 2), "%"
    )
    print(
        "Utilizacion maxima:",
        round(np.max(utilizacion) * 100, 2), "%"
    )

    if len(indices) > 0:
        print(
            "Primer intervalo con utilizacion >= 100%:",
            indices[0] * dt, "s"
        )
    else:
        print("No alcanza el 100% de utilizacion.")

    print("Intervalos con perdidas: no calculados sin buffer.")



# --------------------------------------------------
# 5. Escenario B con buffer de prueba
# --------------------------------------------------

buffer_prueba = 100000.0

resultado_b_buffer = modelo_euler(
    llegadas_b,
    capacidad_b,
    dt,
    buffer_max=buffer_prueba
)

cola_buffer = resultado_b_buffer["cola"]
perdidas_buffer = resultado_b_buffer["perdidas"]

intervalos_perdida = np.where(perdidas_buffer > 0)[0]
intervalos_lleno = np.where(cola_buffer >= buffer_prueba)[0]

print("\nESCENARIO B CON BUFFER DE PRUEBA")
print("-" * 40)
print("Buffer:", buffer_prueba, "paquetes")
print("Cola maxima:", np.max(cola_buffer), "paquetes")
print("Cola final:", cola_buffer[-1], "paquetes")
print("Perdidas estimadas:", np.sum(perdidas_buffer), "paquetes")
print("Intervalos con perdidas:", len(intervalos_perdida))

if len(intervalos_lleno) > 0:
    print(
        "Primer instante con buffer lleno:",
        resultado_b_buffer["tiempo"][intervalos_lleno[0]],
        "s"
    )
else:
    print("El buffer no llega a llenarse.")

if len(intervalos_perdida) > 0:
    print(
        "Primer intervalo con perdidas:",
        intervalos_perdida[0] * dt,
        "s"
    )

print("Nota: el buffer es un supuesto experimental.")



# --------------------------------------------------
# 6. Graficas comparativas
# --------------------------------------------------

fig, ax = plt.subplots(3, 1, figsize=(12, 13))

for nombre, resultado in escenarios.items():

    tiempo = resultado["tiempo"]
    cola = resultado["cola"]
    utilizacion = resultado["utilizacion"]
    llegadas = resultado["llegadas"]

    ax[0].plot(tiempo, cola, label="Escenario " + nombre)
    ax[1].plot(tiempo[:-1], utilizacion, label="Escenario " + nombre)
    ax[2].plot(tiempo[:-1], llegadas, label="Escenario " + nombre)

ax[0].set_title("Evolucion de la cola de paquetes", pad=12)
ax[0].set_ylabel("Paquetes")
ax[0].set_xlabel("Tiempo (s)")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].set_title("Utilizacion del enlace", pad=12)
ax[1].set_ylabel("Utilizacion")
ax[1].set_xlabel("Tiempo (s)")
ax[1].axhline(1.0, linestyle="--", label="Umbral del 100%")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

ax[2].set_title("Tasa de llegada de paquetes", pad=12)
ax[2].set_ylabel("Paquetes/s")
ax[2].set_xlabel("Tiempo (s)")
ax[2].grid(True, alpha=0.3)
ax[2].legend()


fig.subplots_adjust(
    hspace=0.65,
    top=0.95,
    bottom=0.07
)

fig.savefig(
    "resultados_euler.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()