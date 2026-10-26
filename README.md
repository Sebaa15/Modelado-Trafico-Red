# Modelado de trafico de red: Euler, Bayes y Kalman

Proyecto universitario de dinamica de sistemas. Simula la evolucion de colas,
estima estados ruidosos, estudia la incertidumbre y plantea decisiones de capacidad.
**Los escenarios A-D y la variante B con buffer son sinteticos.** No deben
presentarse como mediciones directas de la cola real de CAIDA.

## Instalacion (Windows / VS Code)

Se recomienda Python 3.11 o 3.12. En PowerShell, desde la raiz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecucion

```powershell
.\.venv\Scripts\python.exe main.py
```

Pruebas solamente:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

La ejecucion genera CSV y graficas dentro de `resultados/` y actualiza
`documentacion/kalman.md` a partir de los resultados experimentales.

## Estructura y responsables

- `euler/`: modelo en diferencias y escenarios A-C (integrante 1).
- `modelos/kalman.py` y `modelos/kalman_extendido.py`: estimacion (integrante 3).
- `modelos/prediccion.py` y `modelos/decisiones.py`: pronostico **condicional**
  a llegadas futuras supuestas, alertas y alternativas de capacidad (integrante 3).
- `escenarios/`: escenario D, comparativas, validacion y pronostico (integrante 3).
- `tests/`: verificaciones reproducibles de integracion (integrante 3).
- `documentacion/kalman.md`: metodologia y resultados de Kalman (integrante 3).
- `resultados/`: productos generados por los experimentos.

## Estado del alcance completo

Euler, Kalman, EKF y las comparativas A-D estan implementados.
**Pendiente del integrante 2:** red bayesiana, variables contextuales y tablas
condicionales, obtencion/preparacion de CAIDA y experimento E. Cuando esos modulos
esten disponibles, se debe completar el pipeline Bayes -> Euler -> Kalman -> decisiones.
`main.py` no simula ni sustituye la red bayesiana ausente.

## Consideraciones metodologicas

- `llegadas` y `capacidad` se expresan en paquetes/s; `dt` en segundos;
  `cola` y `perdidas` en paquetes.
- `utilizacion` en Euler representa la relacion de carga ofrecida
  `llegadas/capacidad`; puede superar 1, y no es utilizacion fisica efectiva.
- El buffer limitado introduce una dinamica no lineal. La proyeccion del filtro
  convencional restringe los estados, pero las bandas de covarianza no son una
  garantia de cobertura estadistica bajo saturacion.
- Los escenarios de prediccion usan valores futuros de llegada **supuestos**.
  Sus recomendaciones son comparativas de simulacion, no decisiones optimas
  calculadas con costos reales.

## Reproducibilidad y Git

Las pruebas usan semillas aleatorias fijas; los CSV contienen metricas obtenidas
al ejecutar. Cada integrante realiza commits propios en su rama y solicita una
revision cruzada antes de integrar en `main`.
