# pylint: disable=no-member
"""Detecção de pose e cálculo de ângulos via MediaPipe."""

import math

import cv2
import mediapipe as mp

from src.domain.models import ResultadoAngulo, SnapshotPostural
from src.domain.pontuacao import avaliar_angulo

class AnalisadorDePose:
    """Detecta pontos corporais e avalia os ângulos da modalidade informada."""

    def __init__(self):
        modulo = mp.solutions.pose
        self._pose = modulo.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        self._conexoes = modulo.POSE_CONNECTIONS
        self._enum = modulo.PoseLandmark

    def processar(self, imagem_bgr, snapshot: SnapshotPostural) -> tuple[SnapshotPostural, object]:
        """Preenche o snapshot preservando modalidade, fase e lado recebidos."""

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        resultado = self._pose.process(imagem_rgb)

        if resultado.pose_landmarks is None:
            return snapshot, None

        altura, largura = imagem_bgr.shape[:2]
        landmarks = resultado.pose_landmarks.landmark

        snapshot.pontos = self._extrair_pontos(landmarks, largura, altura, snapshot)
        snapshot.angulos = self._calcular_angulos(snapshot)
        snapshot.pose_detectada = True

        return snapshot, resultado.pose_landmarks

    def _extrair_pontos(self, landmarks, largura: int, altura: int, snapshot: SnapshotPostural) -> dict:
        e = self._enum

        def px(idx):
            return (int(landmarks[idx].x * largura), int(landmarks[idx].y * altura))

        pontos = {
            # Parte superior — lado esquerdo (visível de perfil)
            "orelha":   px(e.LEFT_EAR),
            "ombro":    px(e.LEFT_SHOULDER),
            "cotovelo": px(e.LEFT_ELBOW),
            # Tronco
            "quadril":  px(e.LEFT_HIP),
            "quadril_E": px(e.LEFT_HIP),
            "quadril_D": px(e.RIGHT_HIP),
            # Pernas — ambos os lados
            "joelho_E":    px(e.LEFT_KNEE),
            "tornozelo_E": px(e.LEFT_ANKLE),
            "joelho_D":    px(e.RIGHT_KNEE),
            "tornozelo_D": px(e.RIGHT_ANKLE),
        }

        # Define qual joelho é o principal para análise
        if snapshot.modalidade == "bike":
            pontos["joelho_analise"]    = pontos["joelho_E"]
            pontos["tornozelo_analise"] = pontos["tornozelo_E"]
        else:
            # Corrida: usuário informa qual joelho está à frente
            lado = snapshot.joelho_frente  # "E" ou "D"
            pontos["quadril"] = pontos[f"quadril_{lado}"]
            pontos["joelho_analise"]    = pontos[f"joelho_{lado}"]
            pontos["tornozelo_analise"] = pontos[f"tornozelo_{lado}"]

        return pontos

    def _calcular_angulos(self, snapshot: SnapshotPostural) -> list[ResultadoAngulo]:
        p = snapshot.pontos
        angulos = []

        if snapshot.modalidade == "bike":
            angulos.append(self._avaliar_tronco(p))
            angulos.append(self._avaliar_braco_tronco(p))
            angulos.append(self._avaliar_joelho_bike(p, snapshot.fase_joelho))
        elif snapshot.modalidade == "corrida":
            angulos.append(self._avaliar_joelho_corrida(p, snapshot.fase_joelho))

        return angulos


    def _avaliar_tronco(self, p: dict) -> ResultadoAngulo:
        valor = self._angulo_com_horizontal(p["quadril"], p["ombro"])
        return avaliar_angulo("Tronco", valor, minimo=40.0, maximo=50.0)

    def _avaliar_braco_tronco(self, p: dict) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(p["cotovelo"], p["ombro"], p["quadril"])
        return avaliar_angulo("Braco/Tronco", valor, minimo=85.0, maximo=90.0)

    def _avaliar_joelho_bike(self, p: dict, fase: int) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(p["quadril"], p["joelho_analise"], p["tornozelo_analise"])
        if fase == 1:
            return avaliar_angulo("Joelho fase 1", valor, minimo=68.0)
        return avaliar_angulo("Joelho fase 2", valor, minimo=140.0, maximo=145.0)

    def _avaliar_joelho_corrida(self, p: dict, fase: int) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(p["quadril"], p["joelho_analise"], p["tornozelo_analise"])
        maximo = 160.0 if fase == 1 else 140.0
        return avaliar_angulo(f"Joelho fase {fase}", valor, maximo=maximo)

    @staticmethod
    def _angulo_com_horizontal(ponto_inicial: tuple, ponto_final: tuple) -> float:
        """Calcula o menor ângulo entre um segmento e a horizontal."""
        dx = ponto_final[0] - ponto_inicial[0]
        dy = ponto_final[1] - ponto_inicial[1]
        if dx == 0:
            return 90.0
        angulo = abs(math.degrees(math.atan2(dy, dx)))
        return 180 - angulo if angulo > 90 else angulo

    @staticmethod
    def _angulo_entre_tres_pontos(a: tuple, b: tuple, c: tuple) -> float:
        """Ângulo em B formado pelos segmentos BA e BC."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        produto = ba[0] * bc[0] + ba[1] * bc[1]
        norma_ba = math.hypot(*ba)
        norma_bc = math.hypot(*bc)
        if norma_ba == 0 or norma_bc == 0:
            return 0.0
        cos_ang = max(-1.0, min(1.0, produto / (norma_ba * norma_bc)))
        return math.degrees(math.acos(cos_ang))

    def fechar(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()
