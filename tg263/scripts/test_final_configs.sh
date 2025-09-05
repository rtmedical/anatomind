#!/bin/bash

echo "🧪 Testando configurações de memória corrigidas para Llama-8B na Tesla P40"
echo "==========================================================================="

echo ""
echo "📊 Memória GPU disponível:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Teste 1: TGI com parâmetros ultra-mínimos"
echo "--------------------------------------------"
timeout 90 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --max-batch-total-tokens 1024 \
  --max-total-tokens 1024 \
  --max-input-length 512 \
  --max-batch-size 1 &

TGI_PID=$!
sleep 90
kill $TGI_PID 2>/dev/null
wait $TGI_PID 2>/dev/null
RESULT1=$?

if [ $RESULT1 -eq 0 ] || [ $RESULT1 -eq 143 ]; then
  echo "✅ TGI com parâmetros ultra-mínimos FUNCIONOU!"
else
  echo "❌ TGI com parâmetros ultra-mínimos falhou (exit code: $RESULT1)"
fi

echo ""
echo "🔧 Teste 2: TGI com dtype float16"
echo "---------------------------------"
timeout 90 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --dtype float16 \
  --max-batch-total-tokens 2048 \
  --max-total-tokens 2048 \
  --max-input-length 1024 \
  --max-batch-size 1 &

TGI_PID=$!
sleep 90
kill $TGI_PID 2>/dev/null
wait $TGI_PID 2>/dev/null
RESULT2=$?

if [ $RESULT2 -eq 0 ] || [ $RESULT2 -eq 143 ]; then
  echo "✅ TGI com float16 FUNCIONOU!"
else
  echo "❌ TGI com float16 falhou (exit code: $RESULT2)"
fi

echo ""
echo "🔧 Teste 3: TGI com memória GPU limitada a 80%"
echo "-----------------------------------------------"
timeout 90 docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  rtmedical/anatomind-engine:Llama-8B-TGI \
  --model-id /opt/model \
  --port 3000 \
  --dtype float16 \
  --cuda-memory-fraction 0.8 \
  --max-batch-total-tokens 2048 \
  --max-total-tokens 2048 \
  --max-input-length 1024 \
  --max-batch-size 1 &

TGI_PID=$!
sleep 90
kill $TGI_PID 2>/dev/null  
wait $TGI_PID 2>/dev/null
RESULT3=$?

if [ $RESULT3 -eq 0 ] || [ $RESULT3 -eq 143 ]; then
  echo "✅ TGI com memória limitada FUNCIONOU!"
else
  echo "❌ TGI com memória limitada falhou (exit code: $RESULT3)"
fi

echo ""
echo "🏁 Teste completo!"

# Resultado final
if [ $RESULT1 -eq 0 ] || [ $RESULT1 -eq 143 ]; then
  echo ""
  echo "🎉 CONFIGURAÇÃO VÁLIDA ENCONTRADA!"
  echo "   Use os parâmetros ultra-mínimos no docker-compose.yml"
elif [ $RESULT2 -eq 0 ] || [ $RESULT2 -eq 143 ]; then
  echo ""
  echo "🎉 CONFIGURAÇÃO VÁLIDA ENCONTRADA!"
  echo "   Use dtype float16 no docker-compose.yml"
elif [ $RESULT3 -eq 0 ] || [ $RESULT3 -eq 143 ]; then
  echo ""
  echo "🎉 CONFIGURAÇÃO VÁLIDA ENCONTRADA!"
  echo "   Use memória limitada + float16 no docker-compose.yml"
else
  echo ""
  echo "⚠️  MODELO MUITO GRANDE: Llama-8B não cabe na Tesla P40 (24GB)"
  echo "   SOLUÇÕES:"
  echo "   1. Use um modelo menor (Llama-3B ou 7B)"
  echo "   2. Quantize o modelo externamente antes de usar"
  echo "   3. Use CPU em vez de GPU para este modelo"
fi
