"""Serviço que orquestra o processamento em lote de imagens."""

from pathlib import Path

from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.domain.models import SnapshotPostural
from src.io.imagemBike import carregar_imagem, listar_imagens, salvar_imagem_anotada

def _perguntar_modalidade(numero: int) -> str:
    while True:
        resposta = input(f"\nImagem {numero} — bike (b) ou corrida (c)? ").strip().lower()
        if resposta in ("b", "bike"):
            return "bike"
        if resposta in ("c", "corrida"):
            return "corrida"
        print("  Digite 'b' para bike ou 'c' para corrida.")


def _perguntar_fase_joelho(numero: int, modalidade: str) -> int:
    if modalidade == "bike":
        descricao = "1 - fase superior  |  2 - extensão máxima"
    else:
        descricao = "1 - contato inicial  |  2 - apoio médio"

    while True:
        resposta = input(f"Imagem {numero} — fase do joelho ({descricao}): ").strip()
        if resposta in ("1", "2"):
            return int(resposta)
        print("  Digite 1 ou 2.")


def processar_pasta(pasta_entrada: Path, pasta_saida: Path):
    pasta_saida.mkdir(exist_ok=True)

    imagens = listar_imagens(pasta_entrada)
    if not imagens:
        print("Nenhuma imagem encontrada na pasta de entrada.")
        return

    total = len(imagens)

    print(f"\n{'='*50}")
    print(f"  {total} imagem(ns) encontrada(s). Responda as perguntas:")
    print(f"{'='*50}")

    # Coleta respostas com numeração
    snapshots_configurados = []
    for numero, caminho in enumerate(imagens, start=1):
        modalidade = _perguntar_modalidade(numero)
        fase = _perguntar_fase_joelho(numero, modalidade)
        snapshot = SnapshotPostural(
            caminho_imagem=caminho,
            modalidade=modalidade,
            fase_joelho=fase,
        )
        snapshots_configurados.append((caminho, snapshot))

    print(f"\n{'='*50}")
    print("  Processando imagens...")
    print(f"{'='*50}\n")

    sucesso = 0
    with AnalisadorDePose() as analisador:
        for numero, (caminho, snapshot) in enumerate(snapshots_configurados, start=1):
            imagem = carregar_imagem(caminho)
            if imagem is None:
                print(f"[ERRO] Imagem {numero} — não foi possível abrir: {caminho.name}")
                continue

            snapshot, landmarks = analisador.processar(imagem, snapshot)

            if not snapshot.pose_detectada:
                print(f"[SEM POSE] Imagem {numero} — {caminho.name}")
                continue

            caminho_saida = pasta_saida / f"{caminho.stem}_anotada.jpg"
            salvou = salvar_imagem_anotada(imagem, snapshot, landmarks, caminho_saida)

            if salvou:
                print(f"[OK] Imagem {numero} — {caminho.name} -> {caminho_saida.name}")
                for ang in snapshot.angulos:
                    print(f"     {ang.mensagem}")
                sucesso += 1
            else:
                print(f"[ERRO] Imagem {numero} — não foi possível salvar: {caminho_saida.name}")

    print(f"\nProcessadas com sucesso: {sucesso}/{total}")
    print(f"Saídas salvas em: {pasta_saida}")