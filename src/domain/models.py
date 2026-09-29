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
    desvio_graus: float
    score: float


@dataclass
class SnapshotPostural:
    """Resultado da detecção de pose em uma imagem."""
    caminho_imagem: Path | None = None
    modalidade: str = ""        # "bike" ou "corrida"
    fase_joelho: int = 0        # 1 ou 2
    joelho_frente: str = ""     # "E" ou "D" — só corrida
    pontos: dict = field(default_factory=dict)
    angulos: list = field(default_factory=list)
    pose_detectada: bool = False

    @property
    def score(self) -> float:
        """Média simples dos scores individuais (0 a 1), com quatro casas."""
        if not self.angulos:
            return 0.0
        return round(sum(a.score for a in self.angulos) / len(self.angulos), 4)

    @property
    def resumo(self) -> str:
        dentro = sum(a.dentro_do_padrao for a in self.angulos)
        return f"{dentro} de {len(self.angulos)} angulos dentro do padrao. Score: {self.score:.4f}/1."
