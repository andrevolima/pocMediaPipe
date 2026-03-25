"""Serviço que orquestra o processamento em lote de imagens."""

from pathlib import Path

from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.io.imagemBike import carregar_imagem, listar_imagens, salvar_imagem_anotada


def processar_pasta(pasta_entrada: Path, pasta_saida: Path):
    """Processa todas as imagens da pasta de entrada e salva os resultados."""
    pasta_saida.mkdir(exist_ok=True)

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

            snapshot, landmarks = analisador.processar(imagem, caminho)

            if not snapshot.pose_detectada:
                print(f"[SEM POSE] {caminho.name}")
                continue

            caminho_saida = pasta_saida / f"{caminho.stem}_anotada.jpg"
            salvou = salvar_imagem_anotada(imagem, snapshot, landmarks, caminho_saida)

            if salvou:
                print(f"[OK] {caminho.name} -> {caminho_saida.name}")
                sucesso += 1
            else:
                print(f"[ERRO] Não foi possível salvar: {caminho_saida.name}")

    print(f"\nProcessadas com sucesso: {sucesso}/{total}")
    print(f"Saídas salvas em: {pasta_saida}")