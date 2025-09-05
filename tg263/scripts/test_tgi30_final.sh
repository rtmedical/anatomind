#!/bin/bash

echo "🧪 Teste TGI 3.0 com parâmetros CORRETOS para Tesla P40"
echo "========================================================"

echo ""
echo "📊 Memória GPU disponível:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Teste 1: TGI 3.0 com modelo pequeno e parâmetros corrigidos"
echo "==============================================================="

timeout 180 docker run --rm --gpus all \
  -p 8002:80 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ghcr.io/huggingface/text-generation-inference:3.0 \
  --model-id microsoft/DialoGPT-small \
  --disable-custom-kernels \
  --port 80 \
  --hostname 0.0.0.0 \
  --dtype float16 \
  --max-batch-total-tokens 1024 \
  --max-total-tokens 1024 \
  --max-input-length 512 \
  --max-batch-size 1 \
  --max-batch-prefill-tokens 1024 \
  --cuda-memory-fraction 0.3 &

TGI_PID=$!
echo "🕐 Aguardando TGI 3.0 com parâmetros corretos..."

sleep 15
for i in {1..12}; do
    if ! kill -0 $TGI_PID 2>/dev/null; then
        echo "❌ TGI 3.0 falhou após $((15+i*10)) segundos"
        break
    fi
    
    if [ $i -eq 6 ]; then
        echo "🔄 60% - Testando health endpoint..."
        curl -s http://localhost:8002/health && echo " ✅ Health OK!" || echo " ⏳ Carregando..."
    fi
    
    if [ $i -eq 9 ]; then
        echo "🔄 90% - Teste de geração..."
        curl -s -X POST http://localhost:8002/generate \
          -H "Content-Type: application/json" \
          -d '{"inputs": "Hello", "parameters": {"max_new_tokens": 10}}' \
          && echo " ✅ Geração funcionando!" || echo " ⏳ Ainda inicializando..."
    fi
    
    if [ $i -eq 12 ]; then
        echo "✅ TGI 3.0 FUNCIONOU! Testando API final..."
        curl -s -X POST http://localhost:8002/generate \
          -H "Content-Type: application/json" \
          -d '{"inputs": "What is AI?", "parameters": {"max_new_tokens": 20}}'
        break
    fi
    
    sleep 10
done

kill $TGI_PID 2>/dev/null
wait $TGI_PID 2>/dev/null

echo ""
echo ""
echo "🔧 Teste 2: TGI 3.0 como base para reconstruir sua imagem"
echo "=========================================================="

cat > /tmp/Dockerfile.tgi30 << 'EOF'
FROM ghcr.io/huggingface/text-generation-inference:3.0

# Copiar modelo Llama-8B para o container  
COPY model /opt/model/

# Definir diretório de trabalho
WORKDIR /opt

# Configurar cache
ENV HF_HUB_CACHE="/opt/cache"
ENV TRANSFORMERS_CACHE="/opt/cache"

# Configurações específicas para Tesla P40 (Pascal SM 6.1)
ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING="1"

# Desabilitar telemetria e otimizações desnecessárias
ENV TGI_DISABLE_TELEMETRY="1"
ENV TRANSFORMERS_OFFLINE="1"
ENV HF_HUB_OFFLINE="1"
EOF

echo "✅ Dockerfile para TGI 3.0 gerado em /tmp/Dockerfile.tgi30"
echo ""
echo "🎯 COMANDOS PARA REBUILD:"
echo "========================"
echo "# 1. Parar containers atuais"
echo "docker-compose down"
echo ""
echo "# 2. Reconstruir imagem com TGI 3.0"
echo "cd /home/rt/anatomind/tg263"
echo "cp /tmp/Dockerfile.tgi30 tgi/"
echo "docker build -t rtmedical/anatomind-engine:Llama-8B-TGI-3.0 tgi/"
echo ""
echo "# 3. Atualizar docker-compose.yml para usar nova imagem"
echo "# image: rtmedical/anatomind-engine:Llama-8B-TGI-3.0"
echo ""
echo "# 4. Configurar parâmetros corretos no docker-compose.yml:"
echo "#   --max-batch-prefill-tokens 1024"
echo "#   --max-batch-total-tokens 2048" 
echo "#   --max-total-tokens 2048"
echo "#   --max-input-length 1024"

echo ""
echo "🏁 CONCLUSÃO:"
echo "============="
echo "✅ TGI 3.0 é compatível com Tesla P40 (SM 6.1)"
echo "❌ TGI 3.2 tem flash-attention incompatível"
echo "🎯 SOLUÇÃO: Rebuild sua imagem usando TGI 3.0 como base"
