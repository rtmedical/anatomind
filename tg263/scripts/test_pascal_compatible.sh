#!/bin/bash

echo "🧪 Teste FINAL: TGI com kernels customizados desabilitados para Tesla P40"
echo "========================================================================="

echo ""
echo "📊 Memória GPU atual:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Testando configuração COMPATÍVEL com Pascal (SM 6.1):"
echo ""

# Configuração específica para Tesla P40
timeout 120 docker run --rm --gpus all \
  -p 8001:80 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TGI_DISABLE_TELEMETRY=1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --disable-custom-kernels \
  --port 80 \
  --hostname 0.0.0.0 \
  --dtype float16 \
  --max-batch-total-tokens 1024 \
  --max-total-tokens 1024 \
  --max-input-length 512 \
  --max-batch-size 1 \
  --cuda-memory-fraction 0.75 &

TGI_PID=$!

echo "🕐 Aguardando TGI inicializar (120 segundos)..."
sleep 10

# Verificar se ainda está rodando
for i in {1..11}; do
    if ! kill -0 $TGI_PID 2>/dev/null; then
        echo "❌ TGI falhou após $((i*10)) segundos"
        exit 1
    fi
    
    if [ $i -eq 6 ]; then
        echo "🔄 50% - TGI ainda carregando..."
    fi
    
    if [ $i -eq 9 ]; then
        echo "🔄 80% - Testando endpoint health..."
        curl -s http://localhost:8001/health && echo "✅ Health check OK!" || echo "⏳ Ainda não disponível"
    fi
    
    sleep 10
done

echo ""
echo "🧪 Teste final de API:"
curl -s http://localhost:8001/health && echo "✅ TGI FUNCIONANDO!" || echo "❌ TGI não respondeu"

# Cleanup
kill $TGI_PID 2>/dev/null
wait $TGI_PID 2>/dev/null

echo ""
echo "🏁 Teste completo!"
