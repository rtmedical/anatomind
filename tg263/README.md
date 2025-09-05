# AnatomInd TG-263 API

Sistema Docker offline para servir modelos locais com dois backends (vLLM e TGI) e uma API em Rust compatível com OpenAI, incluindo classificação TG-263.

## 🎯 Características

- **100% Offline**: Sem downloads do Hugging Face Hub
- **Dual Backend**: vLLM para GPUs modernas, TGI para NVIDIA P40 (SM 6.1)
- **API OpenAI Compatible**: Endpoints `/v1/chat/completions` e `/v1/completions`
- **Classificação TG-263**: Endpoint `/classify` com normalização e ranking via Jaro-Winkler
- **Streaming SSE**: Suporte completo para respostas em tempo real
- **Docker Compose**: Orquestração simples de todos os serviços

## 🏗️ Arquitetura

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   vLLM      │    │     TGI      │    │  Rust API   │
│  :8000      │    │    :8001     │    │   :8080     │
│             │    │              │    │             │
│ GPU Modern  │    │  NVIDIA P40  │    │ OpenAI+TG263│
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🚀 Quick Start

### 1. Configuração Inicial

```bash
# Clonar e navegar
cd tg263/

# Copiar configuração
cp .env.example .env

# Editar configurações (opcional)
# Para NVIDIA P40: PROVIDER=tgi
# Para GPUs modernas: PROVIDER=vllm
vim .env
```

### 2. Subir os Serviços

```bash
# Build e start
docker compose up --build

# Ou em background
docker compose up -d --build
```

### 3. Testar a API

```bash
# Tornar executável
chmod +x scripts/*.sh

# Executar exemplos
./scripts/run_curl_examples.sh

# Health check
curl http://localhost:8080/healthz
```

## 📡 Endpoints da API

### Health & Models

```bash
# Status da API
GET /healthz

# Lista de modelos
GET /v1/models
```

### OpenAI Compatible

```bash
# Chat completions
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false,
  "provider": "tgi|vllm"  // opcional
}

# Text completions  
POST /v1/completions
{
  "prompt": "The heart is",
  "max_tokens": 50,
  "stream": true
}
```

### Classificação TG-263

```bash
# Classificar estrutura anatômica
POST /classify
{
  "input": "sphenoid sinus",
  "top_k": 5,
  "provider": "tgi"  // opcional
}

# Resposta
{
  "status": "ok|not_found",
  "mapped_label": "Sinus_Sphenoid",
  "topk": [
    {"label": "Sinus_Sphenoid", "score": 0.95},
    {"label": "VB_T11", "score": 0.23}
  ],
  "raw_output": "Sinus_Sphenoid",
  "provider": "tgi"
}
```

## ⚙️ Configuração

### Variáveis de Ambiente (.env)

```bash
# Provider padrão
PROVIDER=tgi                    # tgi|vllm

# Thresholds de classificação
ABSTAIN_SIM_THRESHOLD=0.85     # 0.0-1.0
MAX_LABELS=5                   # Máximo no top-k
```

### Seleção de Provider

1. **Query Parameter**: `?provider=tgi|vllm`
2. **Request Body**: `{"provider": "tgi|vllm"}`
3. **Environment**: `PROVIDER=tgi|vllm`

## 🖥️ Compatibilidade GPU

### NVIDIA P40 (SM 6.1)
```bash
# Configuração recomendada
PROVIDER=tgi
DISABLE_FLASH_ATTENTION=1

# TGI automaticamente usa:
# --disable-custom-kernels
```

### GPUs Modernas (RTX, A100, etc.)
```bash
# Configuração recomendada  
PROVIDER=vllm

# vLLM com otimizações completas
```

## 🧠 TG-263 Labels

O sistema carrega labels do arquivo `labels/TG263.csv`:

```csv
TG263-Primary Name
Sinus_Sphenoid
VB_T11
A_Subclavian
Lungs
Lig_Hepatogastrc
```

### Processo de Classificação

1. **Geração**: Prompt determinístico (temperature=0)
2. **Normalização**: Minúsculas, sem acentos, alfanumérico
3. **Ranking**: Jaro-Winkler similarity contra labels normalizados
4. **Threshold**: Score >= 0.85 para mapeamento válido
5. **Retorno**: Label oficial ou `not_found`

## 🐳 Docker Images

### Pré-construídas (Oficiais)
- `rtmedical/anatomind-engine:Llama-8B` (vLLM)
- `rtmedical/anatomind-engine:Llama-8B-TGI` (TGI)

### Override de Modelo
```bash
# Colocar pesos em ./model/ para override
# Mount point: /opt/model:ro
```

## 🔧 Troubleshooting

### OOM na GPU
```bash
# Reduzir contexto no TGI
docker compose exec tgi \
  text-generation-launcher --model-id /opt/model \
  --max-total-tokens 4096 \
  --disable-custom-kernels
```

### P40 Flash Attention
```bash
# Já configurado automaticamente
DISABLE_FLASH_ATTENTION=1
```

### Logs
```bash
# Ver logs de todos os serviços
docker compose logs -f

# Serviço específico
docker compose logs -f api
docker compose logs -f vllm
docker compose logs -f tgi
```

## 📊 Monitoramento

### Health Checks
```bash
# API
curl http://localhost:8080/healthz

# vLLM  
curl http://localhost:8000/v1/models

# TGI
curl http://localhost:8001/health
```

### Métricas
- Health checks automáticos no Docker Compose
- Logs estruturados com tracing
- Timeouts configuráveis

## 🛠️ Desenvolvimento

### Build Local
```bash
# Apenas API
cd api/
cargo build --release

# Docker rebuild
docker compose build api
docker compose up api
```

### Adicionar Labels TG-263
```bash
# Editar CSV
vim labels/TG263.csv

# Restart para recarregar
docker compose restart api
```

## 📋 Testes de Aceitação

- ✅ `GET /healthz` retorna `status: ok` e `labels > 0`
- ✅ `GET /v1/models` lista `tg263-8b`
- ✅ `POST /v1/chat/completions` funciona sem streaming
- ✅ `POST /v1/chat/completions` com `stream:true` entrega SSE + `[DONE]`
- ✅ `POST /classify` retorna estrutura completa com scores

## 📄 Licença

Projeto interno RTMedical - Uso restrito
