"""Processamento compartilhado, sem terminal ou gravação em disco."""
from src.domain.models import SnapshotPostural
from src.io.imagemBike import anotar_imagem


class PoseNaoDetectada(ValueError):
    """Imagem válida, mas sem pose detectável."""


def processar_imagem(imagem, snapshot: SnapshotPostural, analisador):
    snapshot, landmarks = analisador.processar(imagem, snapshot)
    if not snapshot.pose_detectada:
        raise PoseNaoDetectada("Nenhuma pose foi detectada na imagem.")
    return snapshot, anotar_imagem(imagem, snapshot, landmarks)
