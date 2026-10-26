"""Genera documentacion/kalman.md con resultados de CSV existentes.

Debe ejecutarse despues de las simulaciones; no inventa metricas ausentes.
"""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "resultados"


def leer_csv(nombre):
    ruta = RES / nombre
    if not ruta.is_file():
        return []
    with ruta.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tabla(filas, columnas, nombres=None):
    if not filas:
        return "*Pendiente: ejecutar el experimento correspondiente.*"
    nombres = nombres or columnas
    encabezado = "| " + " | ".join(nombres) + " |"
    separador = "| " + " | ".join("---" for _ in columnas) + " |"
    def formatear(valor):
        texto = str(valor)
        try:
            numero = float(texto)
        except (TypeError, ValueError):
            return texto
        return str(int(numero)) if numero.is_integer() else f"{numero:.2f}"

    renglones = []
    for fila in filas:
        renglones.append("| " + " | ".join(formatear(fila.get(col, "")) for col in columnas) + " |")
    return "\n".join([encabezado, separador] + renglones)


def generar_informe():
    resumen = leer_csv("resumen_euler_kalman.csv")
    ekf = leer_csv("comparacion_kalman_ekf.csv")
    calib = leer_csv("calibracion_kalman.csv")
    valida = leer_csv("validacion_independiente.csv")
    decisiones = leer_csv("prediccion_decisiones.csv")

    contenidos = [r"""# Estimación del estado de congestión de red con filtros de Kalman

## 1. Objetivo y alcance

Estimar la longitud de la cola de paquetes de un enlace de red a partir de
mediciones con ruido, comparar con el modelo Euler e investigar anticipación
de congestión y decisiones de capacidad. Se implementan el filtro de Kalman
lineal con estado proyectado y una aproximación por tramos de Kalman Extendido
(EKF) para límites del buffer.

Los experimentos de esta sección utilizan **datos sintéticos y parámetros
supuestos**. CAIDA no proporciona directamente la cola interna del router
para validar estas estimaciones.

## 2. Modelo matemático

El modelo usa el estado $x_k=q_k$ y la entrada $a_k=\lambda_k-\mu$:

$$
\hat{x}_{k|k-1}=F\hat{x}_{k-1|k-1}+Ba_k,\qquad
P_{k|k-1}=FP_{k-1|k-1}F^{T}+Q
$$

$$
K_k=P_{k|k-1}H^{T}(HP_{k|k-1}H^{T}+R)^{-1}
$$

$$
\hat{x}_{k|k}=\hat{x}_{k|k-1}+K_k(y_k-H\hat{x}_{k|k-1}),\qquad
P_{k|k}=(I-K_kH)P_{k|k-1}
$$

Para un enlace, $F=1$, $B=\Delta t$, $H=1$. Se usan $\Delta t=1\,s$,
$\mu$ en paquetes/s, $Q$ como varianza del proceso y $R$ como varianza
de medición. El estado simulado por Euler se toma como referencia para medir
errores MAE y RMSE.

## 3. Herramientas y archivos

- `modelos/kalman.py`: filtro con `FilterPy` y proyección al rango físico.
- `modelos/kalman_extendido.py`: EKF escalar con jacobiano por tramos.
- `escenarios/escenario_d.py`: mediciones ruidosas y valores atípicos.
- `escenarios/validacion_kalman.py`: sensibilidad a Q/R y estabilidad de P.
- `escenarios/validacion_independiente.py`: ajuste y prueba en semillas distintas.
- `escenarios/integracion_euler_kalman.py`: comparación de las trayectorias.
- `escenarios/prediccion_decisiones.py`: anticipación de saturación y alternativa de capacidad.

## 4. Experimentos Euler–Kalman (A, B, C y B con buffer)
"""]
    contenidos.append(tabla(resumen, ["Escenario", "Cola maxima", "Perdidas", "MAE mediciones", "MAE Kalman", "RMSE Kalman"]))
    contenidos.append(r"""

El caso A representa un enlace sin acumulación sostenida; B tiene una hora
pico; C reduce la capacidad y B con buffer limita la cola e introduce
pérdidas. La utilización del modelo Euler equivale a **carga ofrecida / capacidad**;
puede ser mayor que el 100 % y no es utilización física del enlace.
El valor de pérdidas cero cuando no hay buffer configurado significa
**pérdidas no modeladas**, no ausencia comprobada de pérdidas reales.

La alta cobertura reportada por algunos experimentos debe interpretarse con
precaución: el filtro se proyecta a estados físicamente posibles y sus bandas
no son intervalos gaussianos rigurosos en los puntos de saturación.

## 5. Kalman frente a EKF
""")
    contenidos.append(tabla(ekf, ["Metodo", "MAE", "RMSE"]))
    contenidos.append(r"""

El EKF modela la transición saturada por segmentos, con jacobiano local igual
a 1 dentro del buffer y aproximado por 0 fuera. En los puntos de cambio de
régimen la derivada no está definida, de modo que las incertidumbres siguen
siendo aproximaciones. No puede afirmarse que EKF es superior en todos los
contextos basándose en un solo escenario.

## 6. Sensibilidad y calibración inicial Q/R
""")
    contenidos.append(tabla(calib[:9], ["Q", "R", "MAE", "RMSE", "Cobertura", "Convergencia"]))
    contenidos.append(r"""

Este barrido inicial usa observaciones sintéticas del mismo experimento para
comparar parámetros. Sirve para comprender sensibilidad, pero la elección
final debe comprobarse fuera de la serie usada para ajustar.

## 7. Calibración con pruebas independientes
""")
    contenidos.append(tabla(valida, ["Q", "R", "RMSE_ajuste", "RMSE_prueba"]))
    contenidos.append(r"""

La selección de parámetros se realiza exclusivamente por `RMSE_ajuste`.
Varias combinaciones con proporciones Q/R similares pueden rendir casi igual;
esto no identifica necesariamente Q y R de forma única.
`RMSE_prueba` se informa para semillas independientes, con ruido de proceso
y medición simulados. Estos resultados tampoco validan tráfico real CAIDA.

## 8. Predicción futura y decisiones de capacidad
""")
    if decisiones:
        contenidos.append(tabla(decisiones, list(decisiones[0].keys())))
    else:
        contenidos.append("*Pendiente: ejecutar `escenarios.prediccion_decisiones`.*")
    contenidos.append(r"""

El estado inicial del pronóstico es la última estimación de Kalman disponible
hasta un instante de corte. El perfil de llegadas del horizonte futuro se
**supone** y no se obtiene de observaciones futuras del filtro. Se compara
la capacidad original con otra capaz de atender la tasa máxima del perfil
supuesto. La comparación es un análisis de escenarios, no una optimización
coste-beneficio ni una regla automática aplicable a routers reales.

## 9. Gráficas y resultados guardados

- `resultados/escenario_d.png`: referencia, observaciones y estimación.
- `resultados/calibracion_kalman.png`: sensibilidad de Q y R.
- `resultados/atipicos_kalman.png`: respuesta ante observaciones anómalas.
- `resultados/integracion_A.png`, `_B.png`, `_C.png`, `_B_buffer.png`: escenarios.
- `resultados/comparacion_kalman_ekf.png`: Kalman/EKF.
- `resultados/validacion_independiente.png`: validación separada.
- `resultados/prediccion_decisiones.png`: alerta y alternativa de capacidad.

## 10. Conclusiones y limitaciones

El enfoque combina un modelo determinista de colas con una estimación que
reduce la influencia del ruido, según las métricas de los escenarios
simulados. Kalman estándar ofrece un método interpretable; EKF permite
aproximar la transición no lineal introducida por un buffer finito. La
predicción ayuda a comparar decisiones antes de alcanzar saturación, bajo
hipótesis explícitas de tráfico futuro.

Las conclusiones son válidas para los escenarios configurados. Falta integrar
las probabilidades contextuales de la red bayesiana y el escenario E con
estadísticas de CAIDA, además de evaluar los resultados completos del sistema.

## 11. Responsabilidad del integrante 3

Implementación de Kalman y EKF; escenario D; calibración, validación y
experimentos comparativos; predicción y decisiones; pruebas de integración;
aporte técnico al informe y defensa oral del sistema completo.
""")
    (BASE / "documentacion").mkdir(exist_ok=True)
    destino = BASE / "documentacion" / "kalman.md"
    destino.write_text("\n".join(contenidos) + "\n", encoding="utf-8")
    print(f"Informe actualizado: {destino}")
    return destino


if __name__ == "__main__":
    generar_informe()
