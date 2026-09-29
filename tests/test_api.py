import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src import api
from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.domain.models import SnapshotPostural
from src.domain.pontuacao import avaliar_angulo


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, 'PASTA_SAIDA', tmp_path / 'saida')
    return TestClient(api.app)


def payload(**updates):
    ok, png = cv2.imencode('.png', np.zeros((200, 200, 3), dtype=np.uint8))
    assert ok
    data = dict(modalidade='bike', fase_joelho=2,
                imagem_base64=base64.b64encode(png).decode())
    data.update(updates)
    return data


class PoseControlada:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def processar(self, imagem, snapshot):
        p = {'orelha': (30, 20), 'ombro': (40, 40), 'cotovelo': (80, 40),
             'quadril': (90, 90), 'quadril_E': (90, 90), 'quadril_D': (95, 90),
             'joelho_E': (110, 120), 'joelho_D': (120, 110),
             'tornozelo_E': (100, 180), 'tornozelo_D': (110, 175)}
        lado = snapshot.joelho_frente if snapshot.modalidade == 'corrida' else 'E'
        p['quadril'] = p[f'quadril_{lado}']
        p['joelho_analise'] = p[f'joelho_{lado}']
        p['tornozelo_analise'] = p[f'tornozelo_{lado}']
        snapshot.pontos = p
        snapshot.pose_detectada = True
        snapshot.angulos = AnalisadorDePose.__new__(AnalisadorDePose)._calcular_angulos(snapshot)
        return snapshot, None


@pytest.mark.parametrize('modalidade,fase,lado,total', [
    ('bike', 1, None, 3), ('bike', 2, None, 3),
    ('corrida', 1, 'E', 1), ('corrida', 2, 'D', 1),
])
def test_sucesso(client, monkeypatch, modalidade, fase, lado, total):
    monkeypatch.setattr(api, 'AnalisadorDePose', PoseControlada)
    response = client.post('/analises', json=payload(modalidade=modalidade, fase_joelho=fase, joelho_frente=lado))
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data['angulos']) == total
    assert 0 <= data['score'] <= 1
    assert data['score'] == pytest.approx(sum(a['score'] for a in data['angulos']) / total, abs=0.0001)
    for angulo in data['angulos']:
        assert angulo['score'] == pytest.approx(max(0, 1 - 0.01 * angulo['desvio_graus']), abs=0.0001)
    image = data['imagem_tratada']
    assert image['arquivo'].startswith('saida/analise_')
    nome = image['arquivo'].split('/')[-1]
    assert (api.PASTA_SAIDA / nome).read_bytes() == base64.b64decode(image['base64'])
    png = cv2.imdecode(np.frombuffer(base64.b64decode(image['base64']), np.uint8), cv2.IMREAD_COLOR)
    assert png.shape[:2] == (image['altura'], image['largura'])
    assert image['altura'] > 200  # painel abaixo da foto
    assert np.any(png[:200, :200])  # marcações na foto
    assert np.any(png[200:] != 25)  # texto no painel


@pytest.mark.parametrize('updates', [dict(modalidade='nadar'), dict(fase_joelho=3),
    dict(fase_joelho=True), dict(fase_joelho='1'), dict(joelho_frente='X'),
    dict(modalidade='corrida'), dict(campo_extra=1), dict(imagem_base64='')])
def test_validacao(client, updates):
    response = client.post('/analises', json=payload(**updates))
    assert response.status_code == 422
    assert 'input' not in response.json()['detail'][0]


@pytest.mark.parametrize('image', ['??', base64.b64encode(b'nao e imagem').decode()])
def test_imagem_invalida(client, image):
    assert client.post('/analises', json=payload(imagem_base64=image)).status_code == 400


def test_limites(client, monkeypatch):
    monkeypatch.setattr(api, 'MAX_IMAGE_BYTES', 1)
    assert client.post('/analises', json=payload()).status_code == 413


def test_limite_pixels(client):
    buf = BytesIO()
    Image.new('1', (5000, 4001)).save(buf, format='PNG')
    assert client.post('/analises', json=payload(imagem_base64=base64.b64encode(buf.getvalue()).decode())).status_code == 413


def test_sem_pose(client, monkeypatch):
    class SemPose(PoseControlada):
        def processar(self, imagem, snapshot):
            return snapshot, None
    monkeypatch.setattr(api, 'AnalisadorDePose', SemPose)
    response = client.post('/analises', json=payload())
    assert response.status_code == 422
    assert 'pose' in response.json()['detail']
    assert not api.PASTA_SAIDA.exists()


def test_arquivos_distintos(client, monkeypatch):
    monkeypatch.setattr(api, 'AnalisadorDePose', PoseControlada)
    respostas = [client.post('/analises', json=payload()) for _ in range(2)]
    assert all(r.status_code == 200 for r in respostas)
    assert respostas[0].json()['imagem_tratada']['arquivo'] != respostas[1].json()['imagem_tratada']['arquivo']
    assert len(list(api.PASTA_SAIDA.glob('*.png'))) == 2


def test_falha_ao_salvar(client, monkeypatch):
    monkeypatch.setattr(api, 'AnalisadorDePose', PoseControlada)
    api.PASTA_SAIDA.write_text('Arquivo impede a criação da pasta', encoding='utf-8')
    resposta = client.post('/analises', json=payload())
    assert resposta.status_code == 500
    assert 'salvar' in resposta.json()['detail']


def test_docs(client):
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json').json()
    endpoint = schema['paths']['/analises']['post']
    assert 'application/json' in endpoint['requestBody']['content']
    assert {'200', '400', '413', '422'} <= endpoint['responses'].keys()
    assert client.get('/health').json() == {'status': 'ok'}
    assert schema['components']['schemas']['AnaliseResponse']['properties']['score']['maximum'] == 1
    assert 'desvio_graus' in schema['components']['schemas']['AnguloResponse']['properties']


def test_score():
    snapshot = SnapshotPostural(angulos=[avaliar_angulo('Tronco', 45, minimo=40, maximo=50),
        avaliar_angulo('Braco/Tronco', 95, minimo=85, maximo=90),
        avaliar_angulo('Joelho', 155, minimo=140, maximo=145)])
    assert snapshot.score == 0.95
    assert '0.9500/1' in snapshot.resumo


def test_quadril_do_lado_informado():
    from types import SimpleNamespace
    import mediapipe as mp
    analisador = AnalisadorDePose.__new__(AnalisadorDePose)
    analisador._enum = mp.solutions.pose.PoseLandmark
    landmarks = [SimpleNamespace(x=i / 100, y=i / 100) for i in range(33)]
    snapshot = SnapshotPostural(modalidade='corrida', fase_joelho=1, joelho_frente='D')
    pontos = analisador._extrair_pontos(landmarks, 100, 100, snapshot)
    assert pontos['quadril'] == pontos['quadril_D']
    assert pontos['quadril'] != pontos['quadril_E']
    assert pontos['joelho_analise'] == pontos['joelho_D']


def test_terminal(tmp_path, monkeypatch, capsys):
    from src.services import analiseService
    entrada = tmp_path / 'entrada'
    entrada.mkdir()
    cv2.imwrite(str(entrada / 'teste.png'), np.zeros((200, 200, 3), dtype=np.uint8))
    respostas = iter(['c', '2', 'D'])
    monkeypatch.setattr('builtins.input', lambda _: next(respostas))
    monkeypatch.setattr(analiseService, 'AnalisadorDePose', PoseControlada)
    analiseService.processar_pasta(entrada, tmp_path / 'saida')
    assert (tmp_path / 'saida' / 'teste_anotada.jpg').exists()
    assert 'Score:' in capsys.readouterr().out
