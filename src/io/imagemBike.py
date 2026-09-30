# pylint: disable=no-member
"""Leitura, anotação e escrita de imagens."""

from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np
from src.domain.models import SnapshotPostural

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def listar_imagens(pasta: Path) -> list[Path]:
    if not pasta.is_dir():
        return []
    return sorted(
        arq for arq in pasta.iterdir()
        if arq.is_file() and arq.suffix.lower() in EXTENSOES_VALIDAS
    )


def carregar_imagem(caminho: Path):
    """Carrega imagem como array BGR. Retorna None se falhar."""
    return cv2.imread(str(caminho))


def _desenhar_arco_e_valor(imagem, vertice, ponto_a, ponto_b, valor, cor, raio=45):
    """Desenha o arco entre dois segmentos e o valor próximo ao vértice."""
    ang_a = math.degrees(math.atan2(ponto_a[1] - vertice[1], ponto_a[0] - vertice[0]))
    ang_b = math.degrees(math.atan2(ponto_b[1] - vertice[1], ponto_b[0] - vertice[0]))

    inicio = min(ang_a, ang_b)
    fim = max(ang_a, ang_b)
    if fim - inicio > 180:
        inicio, fim = fim, inicio + 360

    cv2.ellipse(imagem, vertice, (raio, raio), 0, inicio, fim, cor, 2, cv2.LINE_AA)

    meio = math.radians((inicio + fim) / 2)
    offset = raio + 18
    tx = int(vertice[0] + offset * math.cos(meio))
    ty = int(vertice[1] + offset * math.sin(meio))
    cv2.putText(imagem, f"{valor:.1f}g", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)


def anotar_imagem(
    imagem_original,
    snapshot: SnapshotPostural,
    landmarks_mediapipe=None,
) -> np.ndarray:
    """Desenha os pontos do snapshot e, quando disponível, o esqueleto completo."""
    imagem = imagem_original.copy()
    p = snapshot.pontos
    COR_LINHA = (255, 255, 255)
    COR_PONTO = (0, 255, 0)
    COR_ROTULO = (0, 255, 255)

    # Esqueleto completo do MediaPipe
    if landmarks_mediapipe is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            imagem, landmarks_mediapipe, mp.solutions.pose.POSE_CONNECTIONS,
        )

    # Segmentos principais
    pares = [
        ("orelha", "ombro"), ("ombro", "cotovelo"), ("ombro", "quadril"),
    ]
    for origem, destino in pares:
        cv2.line(imagem, snapshot.pontos[origem], snapshot.pontos[destino], (255, 255, 255), 2)

    # ── Perna esquerda ─────────────────────────────────────────────────────
    cv2.line(imagem, p["quadril_E"],  p["joelho_E"],    COR_LINHA, 2)
    cv2.line(imagem, p["joelho_E"],   p["tornozelo_E"], COR_LINHA, 2)

    # ── Perna direita ──────────────────────────────────────────────────────
    cv2.line(imagem, p["quadril_D"],  p["joelho_D"],    COR_LINHA, 2)
    cv2.line(imagem, p["joelho_D"],   p["tornozelo_D"], COR_LINHA, 2)

    # ── Linha horizontal de referência do tronco (só bike) ─────────────────
    if snapshot.modalidade == "bike":
        ref = (p["quadril"][0] - 120, p["quadril"][1])
        cv2.line(imagem, p["quadril"], ref, (255, 0, 0), 2)

    # ── Pontos e rótulos ───────────────────────────────────────────────────
    rotulos = {
        "orelha":      (5, -10),
        "ombro":       (8, -10),
        "cotovelo":    (-70, -10),
        "quadril":     (8, -8),
        "joelho_E":    (8, -8),
        "tornozelo_E": (8, 15),
        "joelho_D":    (8, -8),
        "tornozelo_D": (8, 15),
    }
    nomes_exibidos = {
        "joelho_E": "joelho E",
        "joelho_D": "joelho D",
        "tornozelo_E": "tornozelo E",
        "tornozelo_D": "tornozelo D",
    }
    for nome, ponto in p.items():
        if nome in ("joelho_analise", "tornozelo_analise", "quadril_E", "quadril_D"):
            continue
        dx, dy = rotulos.get(nome, (8, -8))
        cv2.circle(imagem, ponto, 6, COR_PONTO, -1)
        label = nomes_exibidos.get(nome, nome)
        cv2.putText(imagem, label, (ponto[0] + dx, ponto[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COR_ROTULO, 2)

    # ── Arcos nas articulações ─────────────────────────────────────────────
    for resultado in snapshot.angulos:
        cor = (0, 200, 0) if resultado.dentro_do_padrao else (0, 0, 220)

        if resultado.nome == "Tronco":
            ref = (p["quadril"][0] - 120, p["quadril"][1])
            _desenhar_arco_e_valor(imagem, p["quadril"], p["ombro"], ref,
                                   resultado.valor, cor)

        elif resultado.nome == "Braco/Tronco":
            _desenhar_arco_e_valor(imagem, p["ombro"], p["quadril"], p["cotovelo"],
                                   resultado.valor, cor)

        elif "Joelho" in resultado.nome:
            joelho   = p["joelho_analise"]
            tornozelo = p["tornozelo_analise"]
            _desenhar_arco_e_valor(imagem, joelho, p["quadril"], tornozelo,
                                   resultado.valor, cor)

    # ── Painel de texto ────────────────────────────────────────────────────
    linhas = [f"Modalidade: {snapshot.modalidade.upper()} | Fase: {snapshot.fase_joelho}",
              snapshot.resumo] + [a.mensagem for a in snapshot.angulos]
    largura_texto = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                        for t in linhas) + 30
    largura = max(imagem.shape[1], largura_texto)
    altura_painel = 30 * len(linhas) + 15
    painel = np.full((imagem.shape[0] + altura_painel, largura, 3), 25, dtype=np.uint8)
    painel[:imagem.shape[0], :imagem.shape[1]] = imagem
    imagem = painel
    y = imagem_original.shape[0] + 30
    for linha in linhas[:2]:
        cv2.putText(imagem, linha, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    for resultado in snapshot.angulos:
        cor = (0, 200, 0) if resultado.dentro_do_padrao else (0, 0, 220)
        cv2.putText(imagem, resultado.mensagem, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
        y += 30

    return imagem


def salvar_imagem_anotada(imagem_original, snapshot, landmarks_mediapipe, caminho_saida: Path) -> bool:
    """Compatibilidade com testes manuais em disco."""
    return cv2.imwrite(str(caminho_saida), anotar_imagem(imagem_original, snapshot, landmarks_mediapipe))
