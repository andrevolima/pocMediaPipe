"""Regra de pontuação linear, independente de MediaPipe e da API.

Os limites são inclusivos. Dentro da faixa, o desvio é zero e o score é 1.
Fora dela, cada grau de distância do limite mais próximo desconta 0.01.
Frações de grau também contam; a pontuação nunca fica abaixo de zero.
"""

from src.domain.models import ResultadoAngulo

PENALIDADE_POR_GRAU = 0.01


def calcular_desvio_graus(
    valor: float, minimo: float | None, maximo: float | None,
) -> float:
    """Distância à faixa aceita, e não ao centro dela.

    None indica ausência de limite naquele lado. Por exemplo, para mínimo
    de 68 graus sem máximo, 67 tem desvio 1 e qualquer valor >= 68 tem 0.
    """
    if minimo is None and maximo is None:
        raise ValueError("Informe pelo menos um limite para avaliar o ângulo.")
    if minimo is not None and maximo is not None and minimo > maximo:
        raise ValueError("O limite mínimo não pode exceder o máximo.")
    if minimo is not None and valor < minimo:
        return minimo - valor
    if maximo is not None and valor > maximo:
        return valor - maximo
    return 0.0


def avaliar_angulo(
    nome: str, valor: float, *, minimo: float | None = None, maximo: float | None = None,
) -> ResultadoAngulo:
    """Usa os mesmos limites numéricos para status, score e texto explicativo.

    score = max(0, 1 - desvio_graus * PENALIDADE_POR_GRAU).
    Mantém a precisão interna; o arredondamento ocorre na resposta HTTP.
    """
    desvio = calcular_desvio_graus(valor, minimo, maximo)
    score = max(0.0, 1.0 - desvio * PENALIDADE_POR_GRAU)
    dentro = desvio == 0.0
    if minimo is None:
        ideal = f"<= {maximo:g} graus"
    elif maximo is None:
        ideal = f">= {minimo:g} graus"
    else:
        ideal = f"{minimo:g} - {maximo:g} graus"
    status = "OK" if dentro else "FORA"
    mensagem = (f"{nome}: {valor:.1f} graus {status} (ideal: {ideal}) | "
                f"desvio: {desvio:.2f}g | score: {score:.4f}/1")
    return ResultadoAngulo(nome=nome, valor=valor, ideal=ideal,
                           dentro_do_padrao=dentro, mensagem=mensagem,
                           desvio_graus=desvio, score=score)
