# API de análise postural

Recebe **uma imagem e seus dados em JSON**, processa com MediaPipe e devolve
score, pontos, ângulos, resumo e imagem PNG com marcações e painel da análise.
O fluxo HTTP processa a imagem sem ler o terminal e salva automaticamente o PNG
anotado em `saida/`, na raiz do projeto, antes de retornar a resposta.

## Instalação e execução

Use Python **3.10 a 3.12** e um ambiente virtual:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe main.py
```

- Swagger interativo: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- ReDoc: http://127.0.0.1:8000/redoc
- Disponibilidade: `GET /health`

Alternativa: `.venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000`.
`main.py` também aceita `--host` e `--port`.
O MediaPipe está fixado em 0.10.21 para preservar a API `mp.solutions.pose`.
Na primeira análise, o MediaPipe pode baixar o modelo heavy usado pelo projeto;
nesse caso, é necessário acesso à internet e permissão de escrita no ambiente virtual.

## POST /analises

Envie `Content-Type: application/json`:

```json
{
  "modalidade": "corrida",
  "fase_joelho": 1,
  "joelho_frente": "D",
  "imagem_base64": "BASE64_DA_IMAGEM"
}
```

| Campo | Valores e significado |
|---|---|
| `modalidade` | `bike` ou `corrida` |
| `fase_joelho` | Inteiro 1 ou 2. Bike: 1 = fase superior do pedal; 2 = extensão máxima. Corrida: 1 = contato inicial; 2 = apoio médio. |
| `joelho_frente` | `E` ou `D`, obrigatório para corrida. Opcional na bike, onde não interfere: a análise usa o lado esquerdo. |
| `imagem_base64` | Base64 puro, sem `data:image/...;base64,`. JPEG, PNG, BMP ou WebP; até 10 MiB e 20 milhões de pixels. |

Campos desconhecidos são rejeitados. Envie uma requisição por imagem.
Os exemplos do Swagger usam um marcador: substitua-o por Base64 real.

### Exemplo completo em PowerShell

Com o servidor em execução, em outro terminal:

```powershell
$imagem = Get-ChildItem entrada -File | Select-Object -First 1
$payload = @{
    modalidade = 'bike'
    fase_joelho = 2
    imagem_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagem.FullName))
} | ConvertTo-Json
$resultado = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/analises' -Method Post -ContentType 'application/json' -Body $payload
$resultado.score
$resultado.resumo
[IO.File]::WriteAllBytes((Join-Path $PWD 'resultado_api.png'), [Convert]::FromBase64String($resultado.imagem_tratada.base64))
```

### Resposta 200

Contém `modalidade`, `fase_joelho`, `joelho_frente`, `pose_detectada: true`,
`score`, `resumo`, `angulos`, `pontos` e `imagem_tratada`.
Cada ângulo traz `nome`, `valor` em graus, `ideal`, `dentro_do_padrao`, `mensagem`,
`desvio_graus` e `score` individual (0 a 1).
Os pontos são pares `[x, y]` em pixels da imagem original.
`imagem_tratada` contém `media_type: "image/png"`, `base64`, `largura`, `altura` e `arquivo`.
O campo `arquivo` informa o caminho relativo, por exemplo `saida/analise_<uuid>.png`.
Abra esse arquivo no computador que executa a API para conferir o resultado.
A pasta é criada automaticamente, independentemente do diretório usado para iniciar
o servidor. Cada análise gera um nome único, e o PNG salvo é idêntico ao retornado
em Base64. Os arquivos ficam disponíveis até serem removidos manualmente.
O caminho não é uma URL de download. Se a gravação falhar, a API retorna HTTP 500.
A imagem de saída mantém a foto e acrescenta um painel abaixo; por isso suas dimensões podem aumentar.

### Score e regras

**Versão 2: escala de 0 a 1, quanto maior, melhor.** Substitui o cálculo anterior,
que contava acertos e retornava uma porcentagem de 0 a 100. Consumidores da API
devem atualizar a escala de exibição; o novo score não é a porcentagem de acertos.

1. Dentro da faixa aceita, incluindo seus limites: desvio = 0 e score = 1.
2. Abaixo do mínimo: desvio = mínimo − valor medido.
3. Acima do máximo: desvio = valor medido − máximo.
4. Cada grau de desvio desconta **0,01** do score, até o piso zero.
   Frações de grau também contam: 0,5° de desvio desconta 0,005.

```text
score_do_angulo = max(0, 1 - 0.01 * desvio_graus)
score_geral = round(soma_dos_scores_dos_angulos / quantidade_de_angulos, 4)
```

A distância é medida ao limite mais próximo, não ao centro da faixa.
Quando existe apenas um limite, só há penalização ao ultrapassá-lo na direção
incorreta. A partir de 100° de desvio, o score individual permanece em zero.
Bike usa a média de três ângulos; corrida usa o único ângulo avaliado.
Todos têm o mesmo peso. Primeiro calcula-se a média com precisão completa;
somente na resposta arredondam-se score geral, scores individuais e desvios a
quatro casas decimais. Desvios muito pequenos podem desaparecer no arredondamento;
`dentro_do_padrao` sempre usa o desvio original, sem arredondar.

Exemplos para uma faixa aceita de **140° a 145°**:

| Valor medido | Desvio | Score individual |
|---|---|---|
| 140°, 142° ou 145° | 0° | 1 |
| 139° ou 146° | 1° | 0,99 |
| 139,5° ou 145,5° | 0,5° | 0,995 |
| 135° ou 150° | 5° | 0,95 |
| 130° ou 155° | 10° | 0,90 |
| 40° | 100° | 0 |

Exemplo completo de bike: tronco 45° (score 1), braço/tronco 95° (score 0,95),
joelho fase 2 a 155° (score 0,90). Score geral: `(1 + 0.95 + 0.90) / 3 = 0.95`.

A penalidade de 0,01 por grau é uma decisão de pontuação do produto, definida em
`PENALIDADE_POR_GRAU` no arquivo `src/domain/pontuacao.py`. Não representa
uma calibração clínica, confiança do detector ou diagnóstico clínico.
Nesse arquivo, `calcular_desvio_graus` calcula a distância e `avaliar_angulo`
gera status, score e mensagem a partir dos mesmos limites numéricos.

**Mudança nos limites:** as antigas condições estritas `> 68`, `< 160` e `< 140`
passam a ser `>= 68`, `<= 160` e `<= 140`. Assim, estar exatamente no limite
significa estar dentro do esperado, com desvio zero e score 1.

| Avaliação | Faixa aceita (limites inclusivos) |
|---|---|
| Tronco (bike) | 40 a 50 graus, inclusive |
| Braço/tronco (bike) | 85 a 90 graus, inclusive |
| Joelho bike, fase 1 | Maior ou igual a 68 graus |
| Joelho bike, fase 2 | 140 a 145 graus, inclusive |
| Joelho corrida, fase 1 | Menor ou igual a 160 graus |
| Joelho corrida, fase 2 | Menor ou igual a 140 graus |

Use imagens de perfil com uma pessoa visível. O processamento é de uma pose
estática; não infere modalidade, fase ou lado. Cada requisição cria e fecha sua
instância MediaPipe, sem compartilhar estado entre análises simultâneas.

### Erros

| HTTP | Situação |
|---|---|
| 400 | Base64 inválido, arquivo ilegível ou formato não suportado; `detail` textual |
| 413 | Limite da imagem decodificada em bytes ou pixels excedido; `detail` textual |
| 422 | Dados JSON inválidos, campos ausentes/desconhecidos ou Base64 acima do comprimento permitido; `detail` é lista de erros |
| 422 | Nenhuma pose detectada; `detail` textual, sem score ou imagem tratada |
| 500 | Falha ao codificar ou salvar o PNG na pasta `saida`; `detail` textual |

## Testes manuais pelo terminal

Coloque as imagens em `entrada/` e execute:

```powershell
.venv/Scripts/python.exe main.py --manual
```

As perguntas de modalidade, fase e joelho à frente continuam disponíveis.
As imagens anotadas são salvas em `saida/`, e os ângulos e score aparecem no terminal.
Esse modo usa o mesmo serviço de processamento da API.

## Testes automatizados

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
```

Os testes de contrato usam poses controladas, sem download de modelo.
O contrato de entrada e saída utiliza os [modelos documentados do FastAPI](https://fastapi.tiangolo.com/tutorial/body/).
