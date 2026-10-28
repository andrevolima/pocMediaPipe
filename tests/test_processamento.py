"""Exercita o analisador real com saída controlada do detector MediaPipe."""
from types import SimpleNamespace

import mediapipe as mp
import numpy as np
import pytest
from mediapipe.framework.formats import landmark_pb2

from src.analyzers.poseAnalyzer import AnalisadorDePose
from src.domain.models import SnapshotPostural
from src.services.processamento import processar_imagem


@pytest.mark.parametrize('modalidade,lado,total', [('bike', '', 3), ('corrida', 'D', 1)])
def test_fluxo_preserva_dados_e_desenha_landmarks(modalidade, lado, total):
    landmarks = landmark_pb2.NormalizedLandmarkList()
    for i in range(33):
        landmarks.landmark.add(x=.2 + i * .01, y=.1 + i * .02, z=0, visibility=1)
    analisador = AnalisadorDePose.__new__(AnalisadorDePose)
    analisador._enum = mp.solutions.pose.PoseLandmark
    analisador._pose = SimpleNamespace(process=lambda _: SimpleNamespace(pose_landmarks=landmarks))
    snapshot = SnapshotPostural(modalidade=modalidade, fase_joelho=2, joelho_frente=lado)
    imagem = np.zeros((200, 200, 3), dtype=np.uint8)

    resultado, anotada = processar_imagem(imagem, snapshot, analisador)

    assert resultado is snapshot
    assert (resultado.modalidade, resultado.fase_joelho, resultado.joelho_frente) == (modalidade, 2, lado)
    assert resultado.pose_detectada
    assert len(resultado.angulos) == total
    assert anotada.shape[0] > imagem.shape[0]
    assert np.any(anotada[:200, :200])
    assert not imagem.any()
