"""Exemplos da regra de negócio, limites e penalização progressiva."""
import pytest

from src.domain.models import SnapshotPostural
from src.domain.pontuacao import avaliar_angulo


@pytest.mark.parametrize('valor,desvio,score', [
    (140, 0, 1), (142, 0, 1), (145, 0, 1),
    (139, 1, .99), (146, 1, .99), (139.5, .5, .995), (145.5, .5, .995),
    (135, 5, .95), (150, 5, .95), (130, 10, .90), (155, 10, .90),
    (40, 100, 0), (20, 120, 0),
])
def test_faixa_com_dois_limites(valor, desvio, score):
    resultado = avaliar_angulo('Joelho', valor, minimo=140, maximo=145)
    assert resultado.desvio_graus == desvio
    assert resultado.score == pytest.approx(score)
    assert resultado.dentro_do_padrao == (desvio == 0)


@pytest.mark.parametrize('limites,valor,score', [
    ({'minimo': 68}, 68, 1), ({'minimo': 68}, 90, 1), ({'minimo': 68}, 67, .99),
    ({'maximo': 160}, 160, 1), ({'maximo': 160}, 150, 1), ({'maximo': 160}, 161, .99),
    ({'maximo': 140}, 140, 1), ({'maximo': 140}, 130, 1), ({'maximo': 140}, 145, .95),
])
def test_limite_unilateral(limites, valor, score):
    assert avaliar_angulo('Joelho', valor, **limites).score == pytest.approx(score)


def test_piora_a_cada_grau_ate_o_piso():
    resultados = [avaliar_angulo('Joelho', 145 + d, minimo=140, maximo=145).score
                  for d in range(102)]
    assert all(atual > seguinte for atual, seguinte in zip(resultados[:100], resultados[1:101]))
    assert resultados[100:] == [0, 0]


def test_media_usa_valores_sem_arredondamento_previo():
    angulos = [avaliar_angulo('a', 50.0049, maximo=50),
               avaliar_angulo('b', 50.0049, maximo=50),
               avaliar_angulo('c', 50.0149, maximo=50)]
    assert SnapshotPostural(angulos=angulos).score == .9999


def test_sem_angulos():
    assert SnapshotPostural().score == 0
