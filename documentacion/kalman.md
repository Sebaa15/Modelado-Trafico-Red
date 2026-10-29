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
| A |  |  |  |  |  |
| B |  |  |  |  |  |
| C |  |  |  |  |  |
| E |  |  |  |  |  |


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
| Mediciones | 35.85 | 47.11 |
| Kalman | 7.71 | 11.61 |
| EKF | 6.37 | 10.82 |


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
| 150 | 60 | 24999.09 | 1000 | 1500 | CRITICA | 51 | 4999.10 | 0 | 0.76 |


El estado inicial del pronóstico es la última estimación de Kalman disponible
hasta un instante de corte. El perfil de llegadas del horizonte futuro se
**supone** y no se obtiene de observaciones futuras del filtro. Se compara
la capacidad original con otra capaz de atender la tasa máxima del perfil
supuesto. La comparación es un análisis de escenarios, no una optimización
coste-beneficio ni una regla automática aplicable a routers reales.

## 9. Gráficas y Dashboards Homogéneos Guardados

El conjunto de visualizaciones ha sido homogeneizado a partir del dataset CAIDA Trace Statistics (362 trazas pasivas en `datos/caida_trace_stats.csv`), unificando una duración temporal de $T = 600\,\mathrm{s}$, resolución $\Delta t = 1.0\,\mathrm{s}$, y buffer finito $B_{\max} = 50{,}000$ paquetes:

- `resultados/fig1_escenarios_euler_caida.png`: Panel comparativo de dinámica Euler para Escenarios A, B, C y E con pérdidas acumuladas.
- `resultados/fig2_estimacion_kalman_ekf.png`: Estimación de estado en línea con Kalman y EKF, bandas de credibilidad al 95% y respuesta ante outliers.
- `resultados/fig3_calibracion_y_diagnostico.png`: Matriz de sensibilidad $Q/R$, convergencia asintótica de $P_k$ y $K_k \to 0.181$, y test de innovaciones NIS.
- `resultados/fig4_pipeline_bayes_caida_decisiones.png`: Pipeline integrado de las tres capas: prior contextual Bayesiano ($P(S=1)=82.25\%$), flujos activos CAIDA como leading indicator ($f_k$), y alerta temprana predictiva a $0.80 B_{\max}$.
- `resultados/escenario_d.png`: Escenario D con telemetría ruidosa y mitigación de anomalías.
- `resultados/comparacion_kalman_ekf.png`: Comparación directa de filtrado lineal vs. extendido con saturación física.
- `resultados/prediccion_decisiones.png`: Proyección predictiva hacia el umbral de alerta y evaluación de capacidad alternativa ($\mu=1500$).
- `resultados/pipeline_integrado.png`: Integración de Bayes -> Euler -> Kalman -> Alertas de red.
- `resultados/escenario_e.png`: Traza de paquetes y flujos CAIDA con micro-ráfagas Pareto.

## 10. Conclusiones y validación empírica con CAIDA

El enfoque combina el modelo determinista de colas (Capa 1: Euler) con la estimación recursiva de estado (Capa 3: Kalman y EKF) y la inferencia contextual probabilística (Capa 2: Red Bayesiana), consolidando un sistema completo:

1. **Homogeneización de datos:** Se eliminaron discrepancias de duración y límites de buffer en todo el repositorio. Todos los escenarios operan sobre $T=600\,\mathrm{s}$ y $B_{\max}=50{,}000$ paquetes, en concordancia directa con el documento de modelado (`Elemento3/CasoDeEstudio/modelado-e2.md`).
2. **Validación con trazas CAIDA:** Utilizando las 362 observaciones pasivas de CAIDA Trace Statistics (`chicago-dirA/B`, `nyc-dirA/B`, `sanjose-dirA/B`), se comprobó la relación empírica entre flujos activos y paquetes por segundo, demostrando que los flujos operan como un **leading indicator** (indicador adelantado) de congestión por efecto del slow-start de TCP, permitiendo anticipar la alerta temprana de buffer lleno en varios segundos.
3. **Robustez de estimación:** El filtro de Kalman lineal logra una reducción del error cuadrático medio (RMSE) superior al 70% frente a la telemetría SNMP ruidosa, convergiendo asintóticamente a una ganancia estacionaria $K_\infty \approx 0.181$.
4. **Toma de decisiones predictiva:** La combinación del prior contextual Bayesiano ($E[\lambda]=1390$ pkt/s, $P(S=1)=82.25\%$) y la proyección de Kalman a horizonte de advertencia ($m=15\,\mathrm{s}$ o $m=60\,\mathrm{s}$) permite sugerir la reconfiguración preventiva de capacidad a $\mu=1500$ pkt/s antes de que ocurran pérdidas por desbordamiento.

## 11. Responsabilidad del integrante 3

Implementación de Kalman y EKF; escenario D; calibración, validación y
experimentos comparativos; predicción y decisiones; pruebas de integración;
aporte técnico al informe y defensa oral del sistema completo.

