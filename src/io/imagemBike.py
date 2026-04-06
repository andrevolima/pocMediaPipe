# pylint: disable=no-member
"""Leitura, anotação e escrita de imagens."""

from pathlib import Path

import cv2
import mediapipe as mp
from src.domain.models import SnapshotPostural
import math
import numpy as np

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(
        arq for arq in pasta.iterdir()
        if arq.is_file() and arq.suffix.lower() in EXTENSOES_VALIDAS
    )


def carregar_imagem(caminho: Path):
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


def salvar_imagem_anotada(
    imagem_original,
    snapshot: SnapshotPostural,
    landmarks_mediapipe,      # recebido mas não usado para desenho geral — apenas para consistência
    caminho_saida: Path,
) -> bool:
    imagem = imagem_original.copy()
    p = snapshot.pontos
    COR_LINHA   = (255, 255, 255)
    COR_PONTO   = (0, 255, 0)
    COR_ROTULO  = (0, 255, 255)

    # ── Parte superior: orelha → ombro → cotovelo ──────────────────────────
    cv2.line(imagem, p["orelha"],   p["ombro"],    COR_LINHA, 2)
    cv2.line(imagem, p["ombro"],    p["cotovelo"], COR_LINHA, 2)

    # ── Tronco: ombro → quadril ────────────────────────────────────────────
    cv2.line(imagem, p["ombro"], p["quadril"], COR_LINHA, 2)

    # ── Perna esquerda ─────────────────────────────────────────────────────
    cv2.line(imagem, p["quadril"],    p["joelho_E"],    COR_LINHA, 2)
    cv2.line(imagem, p["joelho_E"],   p["tornozelo_E"], COR_LINHA, 2)

    # ── Perna direita ──────────────────────────────────────────────────────
    cv2.line(imagem, p["quadril"],    p["joelho_D"],    COR_LINHA, 2)
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
        if nome in ("joelho_analise", "tornozelo_analise"):
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
    y = 30
    for resultado in snapshot.angulos:
        cor = (0, 200, 0) if resultado.dentro_do_padrao else (0, 0, 220)
        cv2.putText(imagem, resultado.mensagem, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
        y += 30

    # Modalidade
    largura = imagem.shape[1]
    cv2.putText(imagem, f"Modalidade: {snapshot.modalidade.upper()}", (largura - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.imwrite(str(caminho_saida), imagem)