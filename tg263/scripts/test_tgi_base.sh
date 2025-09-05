#!/bin/bash

echo "🧪 Testando se a imagem base TGI funciona com P40..."

# Teste 1: Tentar rodar TGI base sem modelo (só para ver se inicia)
echo "📦 Teste 1: TGI base sem modelo..."
timeout 60 docker run --rm --gpus all \
  ghcr.io/huggingface/text-generation-inference:3.2 \
  --help

echo ""
echo "📦 Teste 2: Verificar versões na imagem customizada..."
docker run --rm --entrypoint bash \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  -c "echo 'PyTorch:' && python -c 'import torch; print(torch.__version__, torch.version.cuda)' && echo 'CUDA CC support:' && python -c 'import torch; print([f\"SM_{x[0]}{x[1]}\" for x in torch.cuda.get_arch_list() if \".\" not in str(x)])'"

echo ""
echo "📦 Teste 3: Tentar rodar modelo em modo degradado..."
timeout 120 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e DISABLE_FLASH_ATTENTION=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --disable-custom-kernels \
  --max-input-tokens 512 \
  --max-total-tokens 1024 \
  --max-batch-prefill-tokens 256 \
  --max-concurrent-requests 1 \
  --dtype float16 \
  --cuda-memory-fraction 0.5 || echo "❌ Falhou mesmo com configurações mínimas"

echo "✅ Teste completo!"
