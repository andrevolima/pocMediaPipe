"""Estruturas de dados do domínio."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResultadoAngulo:
    """Valor de um ângulo calculado e sua avaliação."""
    nome: str
    valor: float
    ideal: str
    dentro_do_padrao: bool
    mensagem: str


@dataclass
class SnapshotPostural:
    """Resultado da detecção de pose em uma imagem."""
    caminho_imagem: Path
    modalidade: str = ""          # "bike" ou "corrida"
    fase_joelho: int = 0          # 1 ou 2
    pontos: dict = field(default_factory=dict)
    angulos: list = field(default_factory=list)
    pose_detectada: bool = False