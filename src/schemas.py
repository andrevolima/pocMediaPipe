"""Contrato JSON e documentação OpenAPI."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)


class AnaliseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"modalidade": "bike", "fase_joelho": 2, "imagem_base64": "BASE64_DA_IMAGEM"},
        {"modalidade": "corrida", "fase_joelho": 1, "joelho_frente": "D",
         "imagem_base64": "BASE64_DA_IMAGEM"},
    ]})
    modalidade: Literal["bike", "corrida"] = Field(description="Atividade física analisada.")
    fase_joelho: int = Field(strict=True, ge=1, le=2, description=(
        "Bike: 1 = fase superior do pedal; 2 = extensão máxima. "
        "Corrida: 1 = contato inicial da pisada; 2 = apoio médio."))
    joelho_frente: Literal["E", "D"] | None = Field(default=None, description=(
        "Obrigatório na corrida: E = esquerdo, D = direito. "
        "Na bike pode ser omitido e não interfere na análise, que utiliza o lado esquerdo."))
    imagem_base64: str = Field(min_length=1, max_length=MAX_BASE64_LENGTH, description=(
        "JPEG, PNG, BMP ou WebP em Base64 puro, sem prefixo data:. "
        "Máximo de 10 MiB decodificados e 20 milhões de pixels. Uma imagem por requisição."))

    @model_validator(mode="after")
    def validar_corrida(self):
        if self.modalidade == "corrida" and self.joelho_frente is None:
            raise ValueError("joelho_frente é obrigatório para corrida (E ou D).")
        return self


class AnguloResponse(BaseModel):
    nome: str
    valor: float = Field(description="Ângulo em graus.")
    ideal: str
    dentro_do_padrao: bool
    desvio_graus: float = Field(ge=0, description=(
        "Distância em graus até o limite mais próximo da faixa aceita; zero dentro dela. "
        "Limites inclusivos. Arredondado a quatro casas na resposta."))
    score: float = Field(ge=0, le=1, description=(
        "max(0, 1 - 0.01 × desvio_graus). Dentro da faixa: 1; "
        "1 grau fora: 0.99; 10 graus: 0.90; 100 graus ou mais: 0. "
        "Frações de grau contam. Arredondado a quatro casas na resposta."))
    mensagem: str


class ImagemResponse(BaseModel):
    arquivo: str = Field(description=(
        "Caminho do PNG salvo no servidor, relativo à raiz do projeto, "
        "por exemplo saida/analise_<uuid>.png. Não é uma URL de download."))
    media_type: Literal["image/png"] = "image/png"
    base64: str = Field(description="PNG anotado em Base64 puro, incluindo painel de resumo.")
    largura: int
    altura: int


class AnaliseResponse(BaseModel):
    modalidade: Literal["bike", "corrida"]
    fase_joelho: int
    joelho_frente: Literal["E", "D"] | None
    pose_detectada: Literal[True] = True
    score: float = Field(ge=0, le=1, description=(
        "Média simples dos scores individuais, calculada antes do arredondamento e "
        "arredondada a quatro casas. Cada ângulo vale max(0, 1 - 0.01 × desvio em graus). "
        "Bike: três ângulos com pesos iguais; corrida: um. "
        "Não representa confiança do MediaPipe ou diagnóstico clínico."))
    resumo: str
    angulos: list[AnguloResponse]
    pontos: dict[str, tuple[int, int]] = Field(description="Coordenadas [x,y] na imagem original em pixels.")
    imagem_tratada: ImagemResponse


class ErroResponse(BaseModel):
    detail: str
