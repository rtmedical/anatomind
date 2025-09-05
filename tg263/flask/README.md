# API Flask TG-263

API simples para inferência do modelo TG-263, que traduz nomes de estruturas anatômicas para o padrão TG-263.

## Estrutura

```
flesk/
├── Dockerfile              # Container da API Flask
├── src/
│   ├── main.py             # Servidor Flask principal
│   └── inference.py        # Módulo de inferência
├── test_api.py             # Cliente de teste Python
├── test_curl.sh            # Script de teste com curl
└── README.md               # Este arquivo
```

## Uso com Docker Compose

### 1. Construir e iniciar o serviço

```bash
cd /home/rt/anatomind/tg263
docker compose up flask --build
```

### 2. Testar a API

O serviço estará disponível em `http://localhost:5000`

#### Health check
```bash
curl http://localhost:5000/health
```

#### Predição single
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kidney_L"}'
```

#### Predição em lote
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Kidney_L", "Heart", "Liver"]}'
```

#### Obter todas as labels
```bash
curl http://localhost:5000/labels
```

### 3. Scripts de teste

#### Teste com Python
```bash
cd flesk
python test_api.py
```

#### Teste com curl
```bash
cd flesk
./test_curl.sh
```

## Endpoints

### `GET /`
Informações básicas da API.

### `GET /health`
Health check - verifica se o modelo está carregado.

### `POST /predict`
Predição de uma única estrutura anatômica.

**Request:**
```json
{
  "text": "rim esquerdo",
  "min_confidence": -2.5  // opcional
}
```

**Response:**
```json
{
  "output": "Kidney_L",
  "confidence": -1.234,
  "rejected": false
}
```

### `POST /predict_batch`
Predição em lote (máximo 100 itens).

**Request:**
```json
{
  "texts": ["rim esquerdo", "coração", "fígado"],
  "min_confidence": -2.5  // opcional
}
```

**Response:**
```json
{
  "results": [
    {"output": "Kidney_L", "confidence": -1.234, "rejected": false},
    {"output": "Heart", "confidence": -0.987, "rejected": false},
    {"output": "Liver", "confidence": -1.567, "rejected": false}
  ]
}
```

### `GET /labels`
Retorna todas as labels TG-263 disponíveis.

**Response:**
```json
{
  "labels": ["Kidney_L", "Kidney_R", "Heart", "Liver", ...],
  "count": 845
}
```

## Configuração

As seguintes variáveis de ambiente podem ser configuradas:

- `MODEL_PATH`: Caminho para o modelo (padrão: `/workspace/model`)
- `TG_CSV_PATH`: Caminho para o CSV TG-263 (padrão: `/workspace/labels/TG263.csv`)
- `MIN_CONFIDENCE`: Limiar mínimo de confiança (padrão: `-2.5`)
- `HOST`: Host do servidor (padrão: `0.0.0.0`)
- `PORT`: Porta do servidor (padrão: `5000`)

## Volumes Docker

O container mapeia os seguintes volumes:

- `./tgi/model:/workspace/model:ro` - Modelo TG-263
- `./labels:/workspace/labels:ro` - Arquivo CSV com labels TG-263

## GPU

O serviço requer GPU NVIDIA e usa CUDA para acelerar a inferência.

## Logs

Os logs da aplicação mostram:
- Carregamento do modelo
- Predições realizadas
- Erros e exceções

Para ver os logs:
```bash
docker compose logs flask
```
