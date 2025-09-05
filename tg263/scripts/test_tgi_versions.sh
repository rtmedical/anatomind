#!/bin/bash

echo "🧪 Testando diferentes versões TGI para compatibilidade com Tesla P40 (SM 6.1)"
echo "============================================================================="

echo ""
echo "📊 Memória GPU disponível:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "🔧 Teste 1: Verificando Flash Attention no TGI 3.0"
echo "====================================================="

docker run --rm -i --gpus all --entrypoint python \
  ghcr.io/huggingface/text-generation-inference:3.0 <<'PY'
import torch
print("TGI 3.0 - PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)

dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  CC: {cc}")

# Verificar disponibilidade de kernels SDPA
try:
    from torch.backends.cuda import sdp_kernel
    print("SDPA disponível:")
    print("  Flash:", sdp_kernel.is_flash_available())
    print("  Memory Efficient:", sdp_kernel.is_mem_efficient_available()) 
    print("  Math (fallback):", sdp_kernel.is_math_available())
    
    # Forçar apenas math kernel (compatível Pascal)
    sdp_kernel.enable_flash(False)
    sdp_kernel.enable_mem_efficient(False)
    sdp_kernel.enable_math(True)
    print("✅ SDPA configurado para usar apenas 'math' kernel")
    
except Exception as e:
    print("❌ SDPA não disponível:", e)

# Teste básico de atenção
try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    q = torch.randn(1, 8, 128, 64, device=dev)
    k = torch.randn(1, 8, 128, 64, device=dev) 
    v = torch.randn(1, 8, 128, 64, device=dev)
    out = sdpa(q, k, v, is_causal=False)
    print("✅ SDPA math kernel funcionando:", tuple(out.shape))
except Exception as e:
    print("❌ SDPA falhou:", e)
PY

echo ""
echo "🔧 Teste 2: TGI 3.0 com modelo pequeno (compatibilidade)"
echo "=========================================================="

timeout 180 docker run --rm --gpus all \
  -p 8002:80 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e TGI_DISABLE_TELEMETRY=1 \
  ghcr.io/huggingface/text-generation-inference:3.0 \
  --model-id microsoft/DialoGPT-small \
  --disable-custom-kernels \
  --port 80 \
  --hostname 0.0.0.0 \
  --dtype float16 \
  --max-batch-total-tokens 512 \
  --max-total-tokens 512 \
  --max-input-length 256 \
  --max-batch-size 1 \
  --cuda-memory-fraction 0.5 &

TGI_PID=$!
echo "🕐 Aguardando TGI 3.0 inicializar com modelo pequeno..."

sleep 10
for i in {1..15}; do
    if ! kill -0 $TGI_PID 2>/dev/null; then
        echo "❌ TGI 3.0 falhou após $((i*10)) segundos"
        break
    fi
    
    if [ $i -eq 8 ]; then
        echo "🔄 Testando health endpoint..."
        curl -s http://localhost:8002/health && echo " ✅ Health OK!" || echo " ⏳ Ainda carregando..."
    fi
    
    if [ $i -eq 12 ]; then
        echo "🔄 Teste final de API..."
        curl -s http://localhost:8002/health && echo " ✅ TGI 3.0 FUNCIONANDO!" || echo " ❌ Não respondeu"
        break
    fi
    
    sleep 10
done

# Cleanup
kill $TGI_PID 2>/dev/null
wait $TGI_PID 2>/dev/null

echo ""
echo "🔧 Teste 3: TGI 3.0 com sua imagem customizada"
echo "=============================================="

timeout 180 docker run --rm --gpus all \
  -p 8003:80 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST=6.1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TGI_DISABLE_TELEMETRY=1 \
  -e TORCH_BACKENDS_CUDA_ENABLE_FLASH_ATTENTION=0 \
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
  --cuda-memory-fraction 0.7 &

TGI_CUSTOM_PID=$!
echo "🕐 Aguardando sua imagem customizada com TGI..."

sleep 15
for i in {1..12}; do
    if ! kill -0 $TGI_CUSTOM_PID 2>/dev/null; then
        echo "❌ TGI customizado falhou após $((15+i*10)) segundos"
        break
    fi
    
    if [ $i -eq 6 ]; then
        echo "🔄 Verificando saúde da API customizada..."
        curl -s http://localhost:8003/health && echo " ✅ Health OK!" || echo " ⏳ Carregando..."
    fi
    
    if [ $i -eq 10 ]; then
        echo "🔄 Teste final API customizada..."
        curl -s http://localhost:8003/health && echo " ✅ TGI CUSTOMIZADO FUNCIONANDO!" || echo " ❌ Falhou"
        break
    fi
    
    sleep 10
done

# Cleanup
kill $TGI_CUSTOM_PID 2>/dev/null
wait $TGI_CUSTOM_PID 2>/dev/null

echo ""
echo "🏁 Resumo dos Testes:"
echo "===================="
echo "1. TGI 3.0 Flash Attention: Verificado compatibilidade SDPA"
echo "2. TGI 3.0 modelo pequeno: Testado inicialização" 
echo "3. TGI customizado: Testado com flags anti-flash-attention"
echo ""
echo "🎯 PRÓXIMOS PASSOS:"
echo "- Se TGI 3.0 funcionou: rebuild sua imagem com base TGI 3.0"
echo "- Se nenhum funcionou: usar VLLM ou modelo menor"
echo "- Verificar logs detalhados para erros específicos"
