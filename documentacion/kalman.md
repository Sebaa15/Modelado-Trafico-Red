# Estimación del estado de congestión de red con filtros de Kalman

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

| Escenario | Cola maxima | Perdidas | MAE mediciones | MAE Kalman | RMSE Kalman |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 38.93 | 3.43 | 6.18 |
| B | 300000 | 0 | 40.45 | 11.80 | 15.34 |
| C | 240000 | 0 | 39.97 | 12.97 | 16.20 |
| B_buffer | 100000 | 200000 | 39.27 | 8 | 11.96 |


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

| Metodo | MAE | RMSE |
| --- | --- | --- |
| Mediciones | 38.93 | 49.29 |
| Kalman | 8.59 | 12.65 |
| EKF | 7.10 | 11.79 |


El EKF modela la transición saturada por segmentos, con jacobiano local igual
a 1 dentro del buffer y aproximado por 0 fuera. En los puntos de cambio de
régimen la derivada no está definida, de modo que las incertidumbres siguen
siendo aproximaciones. No puede afirmarse que EKF es superior en todos los
contextos basándose en un solo escenario.

## 6. Sensibilidad y calibración inicial Q/R

| Q | R | MAE | RMSE | Cobertura | Convergencia |
| --- | --- | --- | --- | --- | --- |
| 10 | 10000 | 4.05 | 4.91 | 100 | 47 |
| 10 | 2500 | 6.05 | 7.51 | 100 | 23 |
| 100 | 10000 | 7.87 | 9.88 | 100 | 14 |
| 10 | 625 | 9.06 | 11.46 | 85.33 | 12 |
| 100 | 2500 | 11.81 | 15.01 | 99.17 | 7 |
| 1000 | 10000 | 15.18 | 19.36 | 100 | 3 |
| 100 | 625 | 17.25 | 21.92 | 80 | 3 |
| 1000 | 2500 | 21.69 | 27.42 | 98.50 | 2 |
| 1000 | 625 | 28.90 | 36.35 | 74.33 | 1 |


Este barrido inicial usa observaciones sintéticas del mismo experimento para
comparar parámetros. Sirve para comprender sensibilidad, pero la elección
final debe comprobarse fuera de la serie usada para ajustar.

## 7. Calibración con pruebas independientes

| Q | R | RMSE_ajuste | RMSE_prueba |
| --- | --- | --- | --- |
| 16 | 625 | 18.99 | 19.48 |
| 16 | 2500 | 20.28 | 21.25 |
| 16 | 10000 | 25.71 | 26.88 |
| 64 | 625 | 22.04 | 22.35 |
| 64 | 2500 | 18.99 | 19.48 |
| 64 | 10000 | 20.28 | 21.25 |
| 256 | 625 | 28.54 | 28.84 |
| 256 | 2500 | 22.04 | 22.35 |
| 256 | 10000 | 18.99 | 19.48 |


La selección de parámetros se realiza exclusivamente por `RMSE_ajuste`.
Varias combinaciones con proporciones Q/R similares pueden rendir casi igual;
esto no identifica necesariamente Q y R de forma única.
`RMSE_prueba` se informa para semillas independientes, con ruido de proceso
y medición simulados. Estos resultados tampoco validan tráfico real CAIDA.

## 8. Predicción futura y decisiones de capacidad

| corte_s | horizonte_s | cola_estimada_corte | capacidad_actual_pps | capacidad_alternativa_pps | alerta_actual | tiempo_buffer_lleno_desde_corte_s | perdidas_previstas_actual | perdidas_previstas_alternativa | mae_prediccion_vs_euler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 350 | 120 | 75002.10 | 1000 | 1500 | CRITICA | 50 | 35002.11 | 0 | 0.87 |


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

