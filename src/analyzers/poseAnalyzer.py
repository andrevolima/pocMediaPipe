# pylint: disable=no-member
"""Detecção de pose e cálculo de ângulos via MediaPipe."""

import math

import cv2
import mediapipe as mp

from src.domain.models import ResultadoAngulo, SnapshotPostural


class AnalisadorDePose:
    """Detecta pontos corporais e calcula ângulos biomecânicos."""

    def __init__(self):
        modulo = mp.solutions.pose
        self._pose = modulo.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        self._enum = modulo.PoseLandmark

    def processar(self, imagem_bgr, snapshot: SnapshotPostural) -> tuple[SnapshotPostural, object]:
        """Processa a imagem e preenche o snapshot com pontos e ângulos."""
        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        resultado = self._pose.process(imagem_rgb)

        if resultado.pose_landmarks is None:
            return snapshot, None

        altura, largura = imagem_bgr.shape[:2]
        landmarks = resultado.pose_landmarks.landmark

        snapshot.pontos = self._extrair_pontos(landmarks, largura, altura)
        snapshot.angulos = self._calcular_angulos(snapshot)
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

    def _calcular_angulos(self, snapshot: SnapshotPostural) -> list[ResultadoAngulo]:
        """Calcula e avalia todos os ângulos conforme modalidade e fase."""
        pontos = snapshot.pontos
        angulos = []

        if snapshot.modalidade == "bike":
            angulos.append(self._avaliar_tronco(pontos))
            angulos.append(self._avaliar_braco_tronco(pontos))
            angulos.append(self._avaliar_joelho_bike(pontos, snapshot.fase_joelho))

        elif snapshot.modalidade == "corrida":
            angulos.append(self._avaliar_joelho_corrida(pontos, snapshot.fase_joelho))

        return angulos

    # ── Bike ────────────────────────────────────────────────────────────────

    def _avaliar_tronco(self, pontos: dict) -> ResultadoAngulo:
        valor = self._angulo_com_horizontal(pontos["quadril"], pontos["ombro"])
        dentro = 40.0 <= valor <= 50.0
        return ResultadoAngulo(
            nome="Tronco",
            valor=valor,
            ideal="40° – 50°",
            dentro_do_padrao=dentro,
            mensagem=self._formatar_mensagem("Tronco", valor, "40° – 50°", dentro),
        )

    def _avaliar_braco_tronco(self, pontos: dict) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(
            pontos["cotovelo"], pontos["ombro"], pontos["quadril"]
        )
        dentro = 85.0 <= valor <= 90.0
        return ResultadoAngulo(
            nome="Braco/Tronco",
            valor=valor,
            ideal="85° – 90°",
            dentro_do_padrao=dentro,
            mensagem=self._formatar_mensagem("Braco/Tronco", valor, "85° – 90°", dentro),
        )

    def _avaliar_joelho_bike(self, pontos: dict, fase: int) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(
            pontos["quadril"], pontos["joelho"], pontos["tornozelo"]
        )
        if fase == 1:
            dentro = valor > 68.0
            ideal = "> 68°"
        else:
            dentro = 140.0 <= valor <= 145.0
            ideal = "140° – 145°"
        return ResultadoAngulo(
            nome=f"Joelho (fase {fase})",
            valor=valor,
            ideal=ideal,
            dentro_do_padrao=dentro,
            mensagem=self._formatar_mensagem(f"Joelho fase {fase}", valor, ideal, dentro),
        )

    # ── Corrida ─────────────────────────────────────────────────────────────

    def _avaliar_joelho_corrida(self, pontos: dict, fase: int) -> ResultadoAngulo:
        valor = self._angulo_entre_tres_pontos(
            pontos["quadril"], pontos["joelho"], pontos["tornozelo"]
        )
        if fase == 1:
            dentro = valor < 160.0
            ideal = "< 160°"
        else:
            dentro = valor < 140.0
            ideal = "< 140°"
        return ResultadoAngulo(
            nome=f"Joelho corrida (fase {fase})",
            valor=valor,
            ideal=ideal,
            dentro_do_padrao=dentro,
            mensagem=self._formatar_mensagem(f"Joelho fase {fase}", valor, ideal, dentro),
        )

    # ── Geometria ───────────────────────────────────────────────────────────

    @staticmethod
    def _angulo_com_horizontal(ponto_inicial: tuple, ponto_final: tuple) -> float:
        """Ângulo entre um segmento e a horizontal."""
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
        produto_escalar = ba[0] * bc[0] + ba[1] * bc[1]
        norma_ba = math.hypot(*ba)
        norma_bc = math.hypot(*bc)
        if norma_ba == 0 or norma_bc == 0:
            return 0.0
        cos_angulo = max(-1.0, min(1.0, produto_escalar / (norma_ba * norma_bc)))
        return math.degrees(math.acos(cos_angulo))

    @staticmethod
    def _formatar_mensagem(nome: str, valor: float, ideal: str, dentro: bool) -> str:
        simbolo = "OK" if dentro else "FORA"
        return f"{nome}: {valor:.1f}° {simbolo} (ideal: {ideal})"

    def fechar(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()