"""Capa 2: Red Bayesiana de Contexto del Trafico de Red.

Modela P(H, D, A, Lambda, HC, U, CD, Q, S, P) segun el DAG del informe
(Elemento3/CasoDeEstudio/modelado-e2.md).
Provee:
  1. RedBayesianaTrafico: motor de inferencia rapido y directo para simulaciones en linea.
  2. construir_red_bayesiana_pgmpy: modelo formal probabilistico con pgmpy y eliminacion de variables.
"""
from typing import Dict, Tuple, Optional
import numpy as np

try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
    except ImportError:
        from pgmpy.models import BayesianNetwork as BayesianModel
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_DISPONIBLE = True
except ImportError:
    PGMPY_DISPONIBLE = False


class RedBayesianaTrafico:
    """
    Modela P(Lambda, Saturacion | Hora, Dia, Aplicacion, Historial)
    segun el DAG de la Figura 2 y la seccion 1 del informe.
    """
    def __init__(self):
        # Tasas representativas (paquetes/s) para los estados: Baja (800), Media (1000), Alta (1500)
        self.tasas_estados = np.array([800.0, 1000.0, 1500.0])
        self.nombres_estados = ["Baja", "Media", "Alta"]

        # CPT P(Lambda | Hora, Aplicacion) -> [P(Baja), P(Media), P(Alta)]
        self.cpt_lambda = {
            ("Valle", "Web"):       np.array([0.85, 0.12, 0.03]),
            ("Valle", "Streaming"): np.array([0.40, 0.40, 0.20]),
            ("Pico",  "Web"):       np.array([0.30, 0.50, 0.20]),
            ("Pico",  "Streaming"): np.array([0.05, 0.15, 0.80]),
        }

        # CPT P(Saturacion=1 | Estado_Lambda, Historial_Congestion)
        # Estados de Lambda: [Baja (800), Media (1000), Alta (1500)]
        self.cpt_saturacion = {
            "Bajo":     np.array([0.01, 0.20, 0.85]),
            "Moderado": np.array([0.05, 0.40, 0.95]),
            "Alto":     np.array([0.15, 0.65, 0.99]),
        }

    def normalizar_cadena(self, texto: str) -> str:
        """Normaliza aliases comunes para compatibilidad con el informe."""
        t = texto.strip().lower()
        if "valle" in t:
            return "Valle"
        if "pico" in t:
            return "Pico"
        if "stream" in t or "video" in t:
            return "Streaming"
        if "web" in t or "correo" in t:
            return "Web"
        if "bajo" in t:
            return "Bajo"
        if "mod" in t:
            return "Moderado"
        if "alto" in t:
            return "Alto"
        return texto

    def inferir_contexto(self,
                         hora: str,
                         aplicacion: str,
                         historial: str = "Moderado",
                         dia: str = "Laboral") -> Tuple[float, float, np.ndarray]:
        """
        Calcula la tasa esperada E[lambda], la probabilidad marginal de saturacion
        P(S=1 | Contexto) y el vector de probabilidades discretas [P(Baja), P(Media), P(Alta)].

        Parametros:
          hora: 'Valle' o 'Pico'
          aplicacion: 'Web' o 'Streaming'
          historial: 'Bajo', 'Moderado' o 'Alto'
          dia: 'Laboral' o 'FinDeSemana' (informativo)
        """
        h_norm = self.normalizar_cadena(hora)
        a_norm = self.normalizar_cadena(aplicacion)
        hist_norm = self.normalizar_cadena(historial)

        clave = (h_norm, a_norm)
        if clave not in self.cpt_lambda:
            raise ValueError(f"Combinacion (Hora, Aplicacion) desconocida: {clave}")
        if hist_norm not in self.cpt_saturacion:
            raise ValueError(f"Historial de congestion desconocido: {hist_norm}")

        probs_lam = self.cpt_lambda[clave]
        probs_sat_dado_lam = self.cpt_saturacion[hist_norm]

        # Esperanza matematica de la tasa de llegada (paquetes/s)
        lambda_esperada = float(np.sum(probs_lam * self.tasas_estados))

        # Marginalizacion sobre Lambda para obtener P(Saturacion=1 | Contexto)
        prob_saturacion = float(np.sum(probs_lam * probs_sat_dado_lam))

        return lambda_esperada, prob_saturacion, probs_lam

    def consulta_fase3(self) -> Dict[str, float]:
        """
        Resuelve la consulta exacta del entregable cruzado Fase 3:
        Lunes 20:00 (Pico, Laboral), Streaming, Historial Moderado.
        Debe dar P(S=1) = 0.8225 (82.25%) y E[lambda] = 1390.0 pkt/s.
        """
        lam_esp, p_sat, dist = self.inferir_contexto("Pico", "Streaming", "Moderado", "Laboral")
        return {
            "p_baja": float(dist[0]),
            "p_media": float(dist[1]),
            "p_alta": float(dist[2]),
            "lambda_esperada_pps": lam_esp,
            "prob_saturacion": p_sat,
            "prob_saturacion_pct": p_sat * 100.0
        }


def construir_red_bayesiana_pgmpy() -> Optional["BayesianNetwork"]:
    """
    Construye la Red Bayesiana completa segun el DAG con pgmpy:
      Padres de Lambda: Hora (H), Dia (D), Aplicacion (A)
      Padres de Saturacion: Lambda (L), Historial (HC)
      Hojas: Saturacion (S) -> Perdidas (P)
    """
    if not PGMPY_DISPONIBLE:
        return None

    # Definicion de la estructura del grafo
    modelo = BayesianModel([
        ("H", "L"),
        ("A", "L"),
        ("L", "S"),
        ("HC", "S"),
        ("S", "P")
    ])

    # Nodos raiz: H (0: Valle, 1: Pico), A (0: Web, 1: Streaming), HC (0: Bajo, 1: Mod, 2: Alto)
    cpd_h = TabularCPD(variable="H", variable_card=2, values=[[0.6], [0.4]],
                       state_names={"H": ["Valle", "Pico"]})
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]],
                       state_names={"A": ["Web", "Streaming"]})
    cpd_hc = TabularCPD(variable="HC", variable_card=3, values=[[0.4], [0.4], [0.2]],
                        state_names={"HC": ["Bajo", "Moderado", "Alto"]})

    # Lambda condicionado a H y A: (H_card=2, A_card=2 -> 4 columnas)
    # Columnas en orden: (Valle, Web), (Valle, Streaming), (Pico, Web), (Pico, Streaming)
    # Estados de Lambda: [Baja, Media, Alta]
    cpd_l = TabularCPD(
        variable="L",
        variable_card=3,
        values=[
            [0.85, 0.40, 0.30, 0.05],  # P(L=Baja | H, A)
            [0.12, 0.40, 0.50, 0.15],  # P(L=Media | H, A)
            [0.03, 0.20, 0.20, 0.80],  # P(L=Alta | H, A)
        ],
        evidence=["H", "A"],
        evidence_card=[2, 2],
        state_names={
            "L": ["Baja", "Media", "Alta"],
            "H": ["Valle", "Pico"],
            "A": ["Web", "Streaming"]
        }
    )

    # Saturacion condicionada a L (3 estados) y HC (3 estados) -> 9 columnas
    # Columnas ordenadas por (L, HC):
    # (Baja, Bajo), (Baja, Mod), (Baja, Alto),
    # (Media, Bajo), (Media, Mod), (Media, Alto),
    # (Alta, Bajo), (Alta, Mod), (Alta, Alto)
    # P(S=1 | L, HC) del informe:
    # S=1:
    #   L=Baja: Bajo=0.01, Mod=0.05, Alto=0.15
    #   L=Media: Bajo=0.20, Mod=0.40, Alto=0.65
    #   L=Alta: Bajo=0.85, Mod=0.95, Alto=0.99
    p_s1 = [0.01, 0.05, 0.15, 0.20, 0.40, 0.65, 0.85, 0.95, 0.99]
    p_s0 = [1.0 - val for val in p_s1]

    cpd_s = TabularCPD(
        variable="S",
        variable_card=2,
        values=[p_s0, p_s1],
        evidence=["L", "HC"],
        evidence_card=[3, 3],
        state_names={
            "S": ["No", "Si"],
            "L": ["Baja", "Media", "Alta"],
            "HC": ["Bajo", "Moderado", "Alto"]
        }
    )

    # Perdidas P condicionadas a Saturacion S (0: No, 1: Si)
    # Si no hay saturacion (S=No), perdidas casi nulas (0.001)
    # Si hay saturacion (S=Si), alta probabilidad de perdida (0.92)
    cpd_p = TabularCPD(
        variable="P",
        variable_card=2,
        values=[
            [0.999, 0.08],  # P=0 (sin perdidas)
            [0.001, 0.92],  # P=1 (con perdidas)
        ],
        evidence=["S"],
        evidence_card=[2],
        state_names={"P": ["No", "Si"], "S": ["No", "Si"]}
    )

    modelo.add_cpds(cpd_h, cpd_a, cpd_hc, cpd_l, cpd_s, cpd_p)
    assert modelo.check_model(), "Error de consistencia en el modelo probabilistico pgmpy."
    return modelo


def inferir_pgmpy(evidencia: Dict[str, str]) -> Dict[str, float]:
    """Ejecuta inferencia exacta mediante eliminacion de variables."""
    modelo = construir_red_bayesiana_pgmpy()
    if modelo is None:
        raise RuntimeError("pgmpy no esta instalado o no pudo inicializarse.")
    inferencia = VariableElimination(modelo)
    q = inferencia.query(variables=["S"], evidence=evidencia)
    prob_si = float(q.values[1])  # S="Si"
    return {"prob_saturacion": prob_si, "prob_no_saturacion": float(q.values[0])}
