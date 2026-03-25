"""Estruturas de dados do domínio."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SnapshotPostural:
    """Resultado da detecção de pose em uma imagem."""
    caminho_imagem: Path
    pontos: dict = field(default_factory=dict)        # nome -> (x, y)
    angulo_tronco: float = 0.0
    pose_detectada: bool = False