# pylint: disable=no-member
"""Leitura, anotação e escrita de imagens."""

from pathlib import Path

import cv2
import mediapipe as mp

from src.domain.models import SnapshotPostural

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DESLOCAMENTOS_ROTULOS = {
    "orelha":    (-30, -15),
    "ombro":     (10, -15),
    "cotovelo":  (-30, -10),
    "punho":     (-30, -10),
    "quadril":   (10, -5),
    "joelho":    (10, -5),
    "tornozelo": (10, -5),
    "pe":        (10, 15),
}


def listar_imagens(pasta: Path) -> list[Path]:
    """Retorna imagens válidas de uma pasta, ordenadas por nome."""
    return sorted(
        arq for arq in pasta.iterdir()
        if arq.is_file() and arq.suffix.lower() in EXTENSOES_VALIDAS
    )


def carregar_imagem(caminho: Path):
    """Carrega imagem como array BGR. Retorna None se falhar."""
    return cv2.imread(str(caminho))


def salvar_imagem_anotada(
    imagem_original,
    snapshot: SnapshotPostural,
    landmarks_mediapipe,
    caminho_saida: Path,
) -> bool:
    """Anota a imagem com pontos, segmentos e ângulo, e salva em disco."""
    imagem = imagem_original.copy()
    modulo_pose = mp.solutions.pose
    modulo_desenho = mp.solutions.drawing_utils

    # Esqueleto completo do MediaPipe
    modulo_desenho.draw_landmarks(imagem, landmarks_mediapipe, modulo_pose.POSE_CONNECTIONS)

    # Segmentos principais
    pares = [
        ("orelha", "ombro"), ("ombro", "cotovelo"), ("cotovelo", "punho"),
        ("quadril", "joelho"), ("joelho", "tornozelo"), ("tornozelo", "pe"),
    ]
    for origem, destino in pares:
        cv2.line(imagem, snapshot.pontos[origem], snapshot.pontos[destino], (255, 255, 255), 2)

    # Linha de referência horizontal a partir do quadril
    quadril = snapshot.pontos["quadril"]
    cv2.line(imagem, quadril, (quadril[0] - 120, quadril[1]), (255, 0, 0), 2)

    # Pontos e rótulos
    for nome, ponto in snapshot.pontos.items():
        dx, dy = DESLOCAMENTOS_ROTULOS.get(nome, (10, -10))
        cv2.circle(imagem, ponto, 6, (0, 255, 0), -1)
        cv2.putText(imagem, nome, (ponto[0] + dx, ponto[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Informações de ângulo
    cv2.putText(imagem, f"Angulo do tronco: {snapshot.angulo_tronco:.1f} graus",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(imagem, "Linha azul = referencia horizontal",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return cv2.imwrite(str(caminho_saida), imagem)