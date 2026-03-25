# pylint: disable=no-member
"""Detecção de pose e cálculo de ângulos via MediaPipe."""

import math

import cv2
import mediapipe as mp

from src.domain.models import SnapshotPostural


class AnalisadorDePose:
    """Detecta pontos corporais e calcula o ângulo do tronco."""

    def __init__(self):
        modulo = mp.solutions.pose
        self._pose = modulo.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        self._conexoes = modulo.POSE_CONNECTIONS
        self._enum = modulo.PoseLandmark

    def processar(self, imagem_bgr, caminho) -> tuple[SnapshotPostural, object]:
        """Retorna (snapshot, landmarks_mediapipe). landmarks é None se não detectar pose."""
        snapshot = SnapshotPostural(caminho_imagem=caminho)

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        resultado = self._pose.process(imagem_rgb)

        if resultado.pose_landmarks is None:
            return snapshot, None

        altura, largura = imagem_bgr.shape[:2]
        landmarks = resultado.pose_landmarks.landmark

        snapshot.pontos = self._extrair_pontos(landmarks, largura, altura)
        snapshot.angulo_tronco = self._angulo_com_horizontal(
            snapshot.pontos["quadril"],
            snapshot.pontos["ombro"],
        )
        snapshot.pose_detectada = True

        return snapshot, resultado.pose_landmarks

    def _extrair_pontos(self, landmarks, largura: int, altura: int) -> dict:
        """Converte landmarks normalizados em coordenadas de pixel."""
        mapeamento = {
            "orelha":    self._enum.LEFT_EAR,
            "ombro":     self._enum.LEFT_SHOULDER,
            "cotovelo":  self._enum.LEFT_ELBOW,
            "punho":     self._enum.LEFT_WRIST,
            "quadril":   self._enum.LEFT_HIP,
            "joelho":    self._enum.LEFT_KNEE,
            "tornozelo": self._enum.LEFT_ANKLE,
            "pe":        self._enum.LEFT_FOOT_INDEX,
        }
        return {
            nome: (int(landmarks[idx].x * largura), int(landmarks[idx].y * altura))
            for nome, idx in mapeamento.items()
        }

    @staticmethod
    def _angulo_com_horizontal(ponto_inicial: tuple, ponto_final: tuple) -> float:
        """Calcula o menor ângulo entre um segmento e a horizontal."""
        dx = ponto_final[0] - ponto_inicial[0]
        dy = ponto_final[1] - ponto_inicial[1]
        if dx == 0:
            return 90.0
        angulo = abs(math.degrees(math.atan2(dy, dx)))
        return 180 - angulo if angulo > 90 else angulo

    def fechar(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()