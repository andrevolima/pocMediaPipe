# pylint: disable=no-member
"""Leitura, anotação e escrita de imagens."""

from pathlib import Path

import cv2
import mediapipe as mp
from src.domain.models import SnapshotPostural
import math
import numpy as np

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
    return sorted(
        arq for arq in pasta.iterdir()
        if arq.is_file() and arq.suffix.lower() in EXTENSOES_VALIDAS
    )


def carregar_imagem(caminho: Path):
    return cv2.imread(str(caminho))


def _desenhar_arco_e_valor(
    imagem,
    vertice: tuple,
    ponto_a: tuple,
    ponto_b: tuple,
    angulo_graus: float,
    cor: tuple,
    raio: int = 45,
):
    """Desenha o arco entre dois segmentos e o valor do ângulo próximo ao vértice."""
    # Ângulo de cada segmento em relação ao eixo X
    angulo_a = math.degrees(math.atan2(ponto_a[1] - vertice[1], ponto_a[0] - vertice[0]))
    angulo_b = math.degrees(math.atan2(ponto_b[1] - vertice[1], ponto_b[0] - vertice[0]))

    # Garante que o arco vai pelo caminho mais curto
    inicio = min(angulo_a, angulo_b)
    fim = max(angulo_a, angulo_b)

    # Se a diferença for maior que 180°, inverte para pegar o arco menor
    if fim - inicio > 180:
        inicio, fim = fim, inicio + 360

    cv2.ellipse(
        imagem,
        vertice,
        (raio, raio),
        0,
        inicio,
        fim,
        cor,
        2,
        cv2.LINE_AA,
    )

    # Posição do texto: meio do arco, um pouco afastado do vértice
    angulo_meio = math.radians((inicio + fim) / 2)
    offset = raio + 18
    texto_x = int(vertice[0] + offset * math.cos(angulo_meio))
    texto_y = int(vertice[1] + offset * math.sin(angulo_meio))

    cv2.putText(
        imagem,
        f"{angulo_graus:.1f}g",
        (texto_x, texto_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        cor,
        2,
        cv2.LINE_AA,
    )


def salvar_imagem_anotada(
    imagem_original,
    snapshot: SnapshotPostural,
    landmarks_mediapipe,
    caminho_saida: Path,
) -> bool:
    imagem = imagem_original.copy()
    modulo_pose = mp.solutions.pose
    modulo_desenho = mp.solutions.drawing_utils
    p = snapshot.pontos

    # Esqueleto completo do MediaPipe
    modulo_desenho.draw_landmarks(imagem, landmarks_mediapipe, modulo_pose.POSE_CONNECTIONS)

    # Segmentos principais
    pares = [
        ("orelha", "ombro"), ("ombro", "cotovelo"), ("cotovelo", "punho"),
        ("quadril", "joelho"), ("joelho", "tornozelo"), ("tornozelo", "pe"),
    ]
    for origem, destino in pares:
        cv2.line(imagem, p[origem], p[destino], (255, 255, 255), 2)

    # Linha de referência horizontal a partir do quadril (só bike)
    if snapshot.modalidade == "bike":
        cv2.line(imagem, p["quadril"], (p["quadril"][0] - 120, p["quadril"][1]), (255, 0, 0), 2)

    # Pontos e rótulos
    for nome, ponto in p.items():
        dx, dy = DESLOCAMENTOS_ROTULOS.get(nome, (10, -10))
        cv2.circle(imagem, ponto, 6, (0, 255, 0), -1)
        cv2.putText(imagem, nome, (ponto[0] + dx, ponto[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Arcos sobre as articulações
    for resultado in snapshot.angulos:
        cor = (0, 200, 0) if resultado.dentro_do_padrao else (0, 0, 220)

        if resultado.nome == "Tronco":
            # Vértice: quadril | segmentos: quadril→ombro e horizontal
            referencia_horizontal = (p["quadril"][0] - 120, p["quadril"][1])
            _desenhar_arco_e_valor(imagem, p["quadril"], p["ombro"], referencia_horizontal,
                                   resultado.valor, cor)

        elif "Joelho" in resultado.nome:
            # Vértice: joelho | segmentos: joelho→quadril e joelho→tornozelo
            _desenhar_arco_e_valor(imagem, p["joelho"], p["quadril"], p["tornozelo"],
                                   resultado.valor, cor)

        elif "Braco" in resultado.nome:
            # Vértice: ombro | segmentos: ombro→quadril e ombro→cotovelo
            _desenhar_arco_e_valor(imagem, p["ombro"], p["quadril"], p["cotovelo"],
                                   resultado.valor, cor)

    # Painel de resultados no canto superior esquerdo
    y = 35
    for resultado in snapshot.angulos:
        cor = (0, 200, 0) if resultado.dentro_do_padrao else (0, 0, 220)
        cv2.putText(imagem, resultado.mensagem, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
        y += 30

    # Modalidade no canto superior direito
    largura = imagem.shape[1]
    cv2.putText(imagem, f"Modalidade: {snapshot.modalidade.upper()}", (largura - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.imwrite(str(caminho_saida), imagem)