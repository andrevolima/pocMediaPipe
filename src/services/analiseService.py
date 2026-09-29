"""Serviço que orquestra o processamento em lote de imagens."""

from pathlib import Path
import cv2

from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.domain.models import SnapshotPostural
from src.io.imagemBike import carregar_imagem, listar_imagens
from src.services.processamento import PoseNaoDetectada, processar_imagem


def processar_pasta(pasta_entrada: Path, pasta_saida: Path):
    pasta_saida.mkdir(parents=True, exist_ok=True)

    imagens = listar_imagens(pasta_entrada)
    if not imagens:
        print("Nenhuma imagem encontrada na pasta de entrada.")
        return

    total = len(imagens)
    sucesso = 0

    with AnalisadorDePose() as analisador:
        for caminho in imagens:
            imagem = carregar_imagem(caminho)
            if imagem is None:
                print(f"[ERRO] Não foi possível abrir: {caminho.name}")
                continue

            try:
                snapshot, anotada = processar_imagem(imagem, snapshot, analisador)
            except PoseNaoDetectada:
                print(f"[SEM POSE] Imagem {numero} — {caminho.name}")
                continue

            caminho_saida = pasta_saida / f"{caminho.stem}_anotada.jpg"
            salvou = cv2.imwrite(str(caminho_saida), anotada)

            if salvou:
                print(f"[OK] Imagem {numero} — {caminho.name} -> {caminho_saida.name}")
                for ang in snapshot.angulos:
                    print(f"     {ang.mensagem}")
                print(f"     {snapshot.resumo}")
                sucesso += 1
            else:
                print(f"[ERRO] Não foi possível salvar: {caminho_saida.name}")

    print(f"\nProcessadas com sucesso: {sucesso}/{total}")
    print(f"Saidas salvas em: {pasta_saida}")
