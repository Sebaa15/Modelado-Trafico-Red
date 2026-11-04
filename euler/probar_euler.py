import numpy as np
from euler import modelo_euler


def comprobar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


# --------------------------------------------------
# PRUEBA A: trafico moderado
# --------------------------------------------------

llegadas_a = np.full(1200, 800.0)

resultado_a = modelo_euler(
    llegadas_a,
    capacidad=1000.0,
    dt=1.0
)

comprobar(
    np.isclose(np.max(resultado_a["cola"]), 0.0),
    "Escenario A: la cola maxima deberia ser cero."
)

comprobar(
    np.isclose(resultado_a["utilizacion"].max(), 0.8),
    "Escenario A: la utilizacion deberia ser 80%."
)

comprobar(
    np.all(resultado_a["cola"] >= 0),
    "Escenario A: la cola no puede ser negativa."
)


# --------------------------------------------------
# PRUEBA B: pico de trafico durante 10 minutos
# --------------------------------------------------

tiempo_b = np.arange(1200)

llegadas_b = np.full(1200, 800.0)
pico = (tiempo_b >= 200) & (tiempo_b < 800)
llegadas_b[pico] = 1500.0

resultado_b = modelo_euler(
    llegadas_b,
    capacidad=1000.0,
    dt=1.0
)

comprobar(
    np.isclose(np.max(resultado_b["cola"]), 300000.0),
    "Escenario B: la cola maxima esperada es 300000."
)

comprobar(
    np.isclose(resultado_b["cola"][-1], 220000.0),
    "Escenario B: la cola final esperada es 220000."
)

comprobar(
    np.all(resultado_b["cola"] >= 0),
    "Escenario B: la cola no puede ser negativa."
)


# --------------------------------------------------
# PRUEBA C: capacidad reducida
# --------------------------------------------------

llegadas_c = np.full(1200, 800.0)

resultado_c = modelo_euler(
    llegadas_c,
    capacidad=600.0,
    dt=1.0
)

comprobar(
    np.isclose(np.max(resultado_c["cola"]), 240000.0),
    "Escenario C: la cola maxima esperada es 240000."
)

comprobar(
    np.isclose(resultado_c["cola"][-1], 240000.0),
    "Escenario C: la cola final esperada es 240000."
)

comprobar(
    np.all(resultado_c["cola"] >= 0),
    "Escenario C: la cola no puede ser negativa."
)


# --------------------------------------------------
# PRUEBA DEL BUFFER
# --------------------------------------------------

resultado_buffer = modelo_euler(
    llegadas_b,
    capacidad=1000.0,
    dt=1.0,
    buffer_max=100000.0
)

comprobar(
    np.max(resultado_buffer["cola"]) <= 100000.0,
    "La cola supera la capacidad del buffer."
)

comprobar(
    np.isclose(np.sum(resultado_buffer["perdidas"]), 200000.0),
    "Las perdidas estimadas deberian ser 200000."
)

comprobar(
    np.count_nonzero(resultado_buffer["perdidas"] > 0) == 400,
    "Se esperaban 400 intervalos con perdidas."
)


# --------------------------------------------------
# PRUEBA DE VALIDACION DE ENTRADAS
# --------------------------------------------------

entrada_invalida_detectada = False

try:
    modelo_euler(
        np.array([800.0, -20.0, 900.0]),
        capacidad=1000.0,
        dt=1.0
    )
except ValueError:
    entrada_invalida_detectada = True

comprobar(
    entrada_invalida_detectada,
    "El modelo deberia rechazar llegadas negativas."
)


# --------------------------------------------------
# RESULTADO FINAL
# --------------------------------------------------

print("Prueba A superada: trafico moderado.")
print("Prueba B superada: pico de trafico.")
print("Prueba C superada: capacidad reducida.")
print("Prueba de buffer superada.")
print("Prueba de validacion superada.")
print("Todas las pruebas de Euler finalizaron correctamente.")
