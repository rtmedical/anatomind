#!/bin/bash

echo "🧪 Testando otimizações de memória para Llama-8B na Tesla P40"
echo "============================================================"

echo ""
echo "📊 Memória GPU disponível:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Teste 1: TGI com quantização automática INT8"
echo "----------------------------------------------"
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --quantize awq \
  --max-batch-total-tokens 4096 \
  --max-total-tokens 8192 \
  --max-input-length 2048 &

sleep 30
TGI_PID=$!

if kill -0 $TGI_PID 2>/dev/null; then
  echo "✅ TGI com quantização AWQ iniciou com sucesso!"
  kill $TGI_PID
  wait $TGI_PID 2>/dev/null
else
  echo "❌ TGI com quantização AWQ falhou"
fi

echo ""
echo "🔧 Teste 2: TGI com GPTQ quantização"
echo "------------------------------------"
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --quantize gptq \
  --max-batch-total-tokens 2048 \
  --max-total-tokens 4096 \
  --max-input-length 1024 &

sleep 30  
TGI_PID=$!

if kill -0 $TGI_PID 2>/dev/null; then
  echo "✅ TGI com quantização GPTQ iniciou com sucesso!"
  kill $TGI_PID
  wait $TGI_PID 2>/dev/null
else
  echo "❌ TGI com quantização GPTQ falhou"
fi

echo ""
echo "🔧 Teste 3: TGI sem quantização, recursos ultra-baixos"
echo "------------------------------------------------------"
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --max-batch-total-tokens 1024 \
  --max-total-tokens 2048 \
  --max-input-length 512 \
  --max-batch-size 1 &

sleep 45
TGI_PID=$!

if kill -0 $TGI_PID 2>/dev/null; then
  echo "✅ TGI com recursos ultra-baixos iniciou com sucesso!"
  kill $TGI_PID
  wait $TGI_PID 2>/dev/null
else
  echo "❌ TGI com recursos ultra-baixos falhou"
fi

echo ""
echo "🏁 Teste completo de otimizações!"
