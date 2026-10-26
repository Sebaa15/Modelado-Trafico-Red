# Filtro de Kalman y estimación del tráfico de red

## 1. Objetivo

Implementar un filtro de Kalman que permita estimar la cantidad de paquetes acumulados en la cola de un enlace de red a partir de mediciones con ruido, utilizando como referencia el comportamiento generado por el modelo dinámico de Euler.

## 2. Modelo matemático

El filtro utiliza las ecuaciones de espacio de estados:

\[
x_k=Fx_{k-1}+Ba_k+w_k
\]

\[
y_k=Hx_k+v_k
\]

Donde:

- \(x_k\): cantidad de paquetes acumulados en la cola.
- \(F=1\): matriz de transición.
- \(B=\Delta t\): intervalo de muestreo.
- \(a_k=\lambda_k-\mu\): diferencia entre la tasa de llegada y la capacidad de servicio.
- \(H=1\): matriz de observación.
- \(Q\): covarianza del ruido del proceso.
- \(R\): covarianza del ruido de medición.

El algoritmo realiza dos operaciones principales: predicción del estado mediante el modelo dinámico y corrección de la estimación utilizando una nueva medición.

## 3. Implementación

Se utilizó Python junto con NumPy y FilterPy.

El archivo `modelos/kalman.py` contiene el algoritmo de estimación. El archivo `escenarios/escenario_d.py` permite comprobar su funcionamiento utilizando mediciones sintéticas con ruido.

Se consideraron inicialmente los parámetros experimentales:

| Parámetro | Valor |
|---|---|
| Intervalo de muestreo | 1 segundo |
| Capacidad de referencia | 1000 paquetes/s |
| Q inicial | 100 |
| R inicial | 2500 |
| Duración del escenario D | 600 segundos |

Estos parámetros se someten a pruebas de sensibilidad y calibración.

## 4. Validación experimental

Se realizaron o se prepararon las siguientes pruebas:

1. Comparación entre cola simulada, mediciones ruidosas y estimaciones de Kalman.
2. Evaluación del error mediante MAE y RMSE.
3. Calibración de los parámetros Q y R.
4. Análisis de la convergencia de la incertidumbre.
5. Evaluación ante mediciones atípicas.
6. Integración con los escenarios generados mediante Euler.

## 5. Integración con Euler

Se utiliza el estado de la cola calculado por Euler como referencia sintética. A partir de esta serie se generan mediciones con ruido, que son procesadas por Kalman.

Se consideran los escenarios de tráfico moderado, hora pico, capacidad reducida y buffer limitado.

Para los escenarios con restricciones físicas, la estimación se limita al intervalo permitido para la cola. Esta proyección evita estados físicamente imposibles, aunque no constituye por sí misma un filtro de Kalman extendido y requiere precaución al interpretar su incertidumbre.

## 6. Resultados

Se presentarán las gráficas comparativas y una tabla de errores MAE y RMSE para cada escenario.

**Resultados numéricos:** completar con los valores obtenidos de `resultados/resumen_euler_kalman.csv` y `resultados/calibracion_kalman.csv`.

## 7. Interpretación y limitaciones

La precisión del filtro se evalúa comparando sus estimaciones con los estados generados por Euler. Una reducción de MAE y RMSE frente a las mediciones originales constituye evidencia de mejora en el experimento correspondiente.

La estimación utiliza datos sintéticos para su validación, por lo que estos resultados no deben interpretarse como mediciones directas de la cola interna de una red CAIDA.

Para escenarios no lineales, la evaluación puede ampliarse mediante EKF o UKF, conforme a las extensiones contempladas en el documento del proyecto.
