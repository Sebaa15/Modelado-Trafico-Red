# Modelado de trafico de red: Euler, Bayes y Kalman

Proyecto universitario de dinamica de sistemas. Simula la evolucion de colas,
estima estados ruidosos, estudia la incertidumbre y plantea decisiones de capacidad.
Los escenarios A-D y la variante B con buffer son sinteticos; el escenario E modela
propiedades auto-similares de trazas troncales tipo CAIDA.

## Instalacion (Windows / VS Code)

Se recomienda Python 3.11 o 3.12. En PowerShell, desde la raiz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecucion

Ejecutar la suite completa (13 actividades: pruebas, simulaciones y graficos):

```powershell
.\.venv\Scripts\python.exe main.py
```

Pruebas automatizadas solamente (17 verificaciones unitarias y de integracion):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

La ejecucion genera CSV y graficas dentro de `resultados/` y actualiza
`documentacion/kalman.md` a partir de los resultados experimentales.

## Estructura y responsables

- `euler/`: modelo en diferencias y escenarios A-C con buffer finito (integrante 1).
- `modelos/red_bayesiana.py`: red bayesiana contextual, DAG, tablas CPT e inferencia (integrante 2).
- `modelos/kalman.py` y `modelos/kalman_extendido.py`: estimacion en linea y filtros lineal y EKF (integrante 3).
- `modelos/prediccion.py` y `modelos/decisiones.py`: pronostico condicional, alertas y alternativas de capacidad.
- `escenarios/inferencia_bayesiana.py`: consulta cruzada Fase 3 y analisis de sensibilidad contextual (integrante 2).
- `escenarios/escenario_e.py`: traza backbone CAIDA, auto-similitud, calibracion de $\mu$ y flujos como leading indicator (integrante 2).
- `escenarios/pipeline_integrado.py`: pipeline colaborativo cerrado Bayes -> Euler -> Kalman -> Decisiones (integracion completa).
- `escenarios/`: escenario D, comparativas Kalman/EKF, validacion independiente.
- `tests/`: 17 verificaciones automatizadas de todo el sistema integrado.
- `documentacion/kalman.md`: metodologia y resultados de Kalman actualizados.
- `resultados/`: productos generados (PNG y CSV) para analisis e informes.

## Estado del alcance completo

Todas las capas y escenarios del informe (`Elemento3/CasoDeEstudio/modelado-e2.md`) estan implementados e integrados:
- **Capa 1 (Dinamica Determinista):** Modelo de Euler con buffer finito $B_{\max} = 50{,}000$ paquetes y cuantificacion de perdidas (*tail-drop*) para los Escenarios A, B y C.
- **Capa 2 (Modelo Contextual):** Red Bayesiana con motor probabilistico directo y formal en `pgmpy` (DAG de 10 variables, consulta Fase 3 con riesgo de saturacion exacto de 82.25% y $E[\lambda] = 1390$ pkt/s).
- **Capa 3 (Estimacion de Estado en Linea):** Filtro de Kalman lineal con proyeccion fisica y EKF escalar por tramos; reduccion de ruido superior al 75% frente a telemetria ruidosa y bandas de credibilidad al 95%.
- **Escenario E (Trafico Real CAIDA):** Series temporales de paquetes/s ($\lambda_k$) y flujos/s ($f_k$) con rafagas Pareto auto-similares, calibracion de $\mu$ al percentil 99.9 y anticipacion temprana gracias a los flujos como leading indicator.
- **Pipeline Integral:** Bayes alimenta a Euler, Euler estructura el paso predictivo a priori de Kalman, Kalman proyecta el horizonte de decision y emite alertas de saturacion con decenas de segundos de anticipacion preventiva.

## Consideraciones metodologicas

- `llegadas` y `capacidad` se expresan en paquetes/s; `dt` en segundos;
  `cola` y `perdidas` en paquetes.
- `utilizacion` en Euler representa la relacion de carga ofrecida
  `llegadas/capacidad`; puede superar 1, y no es utilizacion fisica efectiva.
- El buffer limitado introduce una dinamica no lineal. La proyeccion del filtro
  convencional restringe los estados, pero las bandas de covarianza no son una
  garantia de cobertura estadistica bajo saturacion.
- Los escenarios de prediccion usan valores futuros de llegada supuestos o inferidos
  contextualmente.

## Reproducibilidad y Git

Las pruebas usan semillas aleatorias fijas; los CSV contienen metricas obtenidas
al ejecutar. Cada integrante realiza commits propios en su rama y solicita una
revision cruzada antes de integrar en `main`.
