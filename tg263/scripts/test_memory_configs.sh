#!/bin/bash

echo "🧪 Testando configurações corretas de memória para Llama-8B na Tesla P40"
echo "========================================================================"

echo ""
echo "📊 Memória GPU disponível:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Teste 1: TGI com parâmetros mínimos válidos"
echo "----------------------------------------------"
timeout 60 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --max-batch-total-tokens 2048 \
  --max-total-tokens 2048 \
  --max-input-length 1024 \
  --max-batch-size 1 \
  --cuda-graphs ""

RESULT1=$?
if [ $RESULT1 -eq 124 ]; then
  echo "✅ TGI com parâmetros mínimos iniciou (timeout após 60s = sucesso)"
elif [ $RESULT1 -eq 0 ]; then
  echo "✅ TGI com parâmetros mínimos finalizou normalmente"
else
  echo "❌ TGI com parâmetros mínimos falhou (exit code: $RESULT1)"
fi

echo ""
echo "🔧 Teste 2: TGI com dtype float16 (menos memória)"
echo "------------------------------------------------"
timeout 60 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --dtype float16 \
  --max-batch-total-tokens 4096 \
  --max-total-tokens 4096 \
  --max-input-length 2048 \
  --max-batch-size 1 \
  --cuda-graphs ""

RESULT2=$?
if [ $RESULT2 -eq 124 ]; then
  echo "✅ TGI com float16 iniciou (timeout após 60s = sucesso)"
elif [ $RESULT2 -eq 0 ]; then
  echo "✅ TGI com float16 finalizou normalmente"
else
  echo "❌ TGI com float16 falhou (exit code: $RESULT2)"
fi

echo ""
echo "🔧 Teste 3: TGI com cuda_memory_fraction reduzida"
echo "------------------------------------------------"
timeout 60 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --dtype float16 \
  --cuda-memory-fraction 0.8 \
  --max-batch-total-tokens 4096 \
  --max-total-tokens 4096 \
  --max-input-length 2048 \
  --max-batch-size 1 \
  --cuda-graphs ""

RESULT3=$?
if [ $RESULT3 -eq 124 ]; then
  echo "✅ TGI com memória reduzida iniciou (timeout após 60s = sucesso)"
elif [ $RESULT3 -eq 0 ]; then
  echo "✅ TGI com memória reduzida finalizou normalmente"
else
  echo "❌ TGI com memória reduzida falhou (exit code: $RESULT3)"
fi

echo ""
echo "🏁 Teste completo de configurações!"

# Resultado final
if [ $RESULT1 -eq 124 ] || [ $RESULT2 -eq 124 ] || [ $RESULT3 -eq 124 ]; then
  echo ""
  echo "🎉 SUCESSO: Pelo menos uma configuração funcionou!"
  echo "   Use os parâmetros da configuração que teve sucesso no docker-compose.yml"
elif [ $RESULT1 -eq 0 ] || [ $RESULT2 -eq 0 ] || [ $RESULT3 -eq 0 ]; then
  echo ""
  echo "🎉 SUCESSO: Pelo menos uma configuração finalizou sem erro!"
else
  echo ""
  echo "⚠️  PROBLEMA: O modelo Llama-8B é muito grande para a Tesla P40 (24GB)"
  echo "   Considere usar um modelo menor (3B ou 7B) ou quantizar o modelo"
fi
