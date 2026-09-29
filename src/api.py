"""API HTTP com processamento isolado por requisição."""
import base64
import binascii
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.domain.models import SnapshotPostural
from src.schemas import AnaliseRequest, AnaliseResponse, ErroResponse, MAX_IMAGE_BYTES
from src.services.processamento import PoseNaoDetectada, processar_imagem

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "saida"

app = FastAPI(title="API de análise postural", version="2.0.0", description=(
    "Recebe metadados e uma imagem em JSON, analisa com MediaPipe e retorna "
    "score, ângulos, pontos e PNG anotado em Base64. Salva o mesmo PNG na pasta saida "
    "do projeto, com nome único informado em imagem_tratada.arquivo. "
    "Score de 0 a 1: cada ângulo começa em 1 e perde 0.01 por grau fora da faixa, "
    "até zero. O score geral é a média dos ângulos, com pesos iguais. "
    "Todos os limites são inclusivos: tronco bike [40,50], braço/tronco [85,90], "
    "joelho bike fase 1 >=68 e fase 2 [140,145]; corrida fase 1 <=160 e fase 2 <=140. "
    "Exemplo: joelho bike fase 2 a 150 graus tem desvio 5 e score 0.95. "
    "Versão 2: substitui a antiga porcentagem de acertos (0 a 100). "
    "O score não é um diagnóstico clínico."))


@app.exception_handler(RequestValidationError)
async def erro_validacao(request: Request, exc: RequestValidationError):
    # Não devolver a imagem inteira nos erros de validação.
    erros = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": erros})


def decodificar_imagem(conteudo: str):
    try:
        dados = base64.b64decode(conteudo, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, "imagem_base64 deve conter Base64 puro válido.") from exc
    if len(dados) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "A imagem excede 10 MiB.")
    try:
        with Image.open(BytesIO(dados)) as imagem:
            if imagem.format not in {"JPEG", "PNG", "BMP", "WEBP"}:
                raise HTTPException(400, "Formato não suportado. Use JPEG, PNG, BMP ou WebP.")
            if imagem.width * imagem.height > 20_000_000:
                raise HTTPException(413, "A imagem excede 20 milhões de pixels.")
            imagem.verify()
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "A imagem excede o limite de pixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(400, "Os dados não representam uma imagem válida.") from exc
    try:
        imagem = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise HTTPException(400, "Não foi possível decodificar a imagem.") from exc
    if imagem is None:
        raise HTTPException(400, "Não foi possível decodificar a imagem.")
    return imagem


@app.get("/health", tags=["Serviço"], summary="Verifica disponibilidade da API")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analises", response_model=AnaliseResponse, tags=["Análise"],
          summary="Analisa uma imagem de bike ou corrida",
          responses={
              400: {"model": ErroResponse, "description": "Base64 ou imagem inválida / formato não suportado."},
              413: {"model": ErroResponse, "description": "Imagem acima do limite de bytes ou pixels."},
              500: {"model": ErroResponse, "description": "Falha ao codificar ou salvar a imagem tratada."},
              422: {"description": "Metadados inválidos (detail: lista) ou pose não detectada (detail: texto).",
                    "content": {"application/json": {"schema": {"type": "object", "properties": {
                        "detail": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]}
                    }}}}},
          })
def analisar(dados: AnaliseRequest):
    """No Swagger, substitua BASE64_DA_IMAGEM pelo conteúdo real.
    Sem pose detectada, retorna 422 sem atribuir score.
    """
    imagem = decodificar_imagem(dados.imagem_base64)
    snapshot = SnapshotPostural(modalidade=dados.modalidade, fase_joelho=dados.fase_joelho,
                               joelho_frente=dados.joelho_frente or "")
    try:
        with AnalisadorDePose() as analisador:
            snapshot, anotada = processar_imagem(imagem, snapshot, analisador)
    except PoseNaoDetectada as exc:
        raise HTTPException(422, str(exc)) from exc
    sucesso, png = cv2.imencode(".png", anotada)
    if not sucesso:
        raise HTTPException(500, "Falha ao codificar a imagem tratada.")
    # Salva os mesmos bytes devolvidos em Base64, sem recodificar a imagem.
    # A pasta é relativa ao projeto, independentemente do diretório de execução.
    nome_arquivo = f"analise_{uuid4().hex}.png"
    png_bytes = png.tobytes()
    try:
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        with (PASTA_SAIDA / nome_arquivo).open("xb") as arquivo:
            arquivo.write(png_bytes)
    except OSError as exc:
        raise HTTPException(500, "Não foi possível salvar a imagem tratada na pasta saida.") from exc
    return {
        "modalidade": dados.modalidade, "fase_joelho": dados.fase_joelho,
        "joelho_frente": dados.joelho_frente, "pose_detectada": True,
        "score": snapshot.score, "resumo": snapshot.resumo,
        "angulos": [{**asdict(a), "score": round(a.score, 4),
                     "desvio_graus": round(a.desvio_graus, 4)} for a in snapshot.angulos],
        "pontos": snapshot.pontos,
        "imagem_tratada": {"media_type": "image/png", "base64": base64.b64encode(png_bytes).decode("ascii"),
                           "arquivo": f"saida/{nome_arquivo}",
                           "largura": anotada.shape[1], "altura": anotada.shape[0]},
    }
