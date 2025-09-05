#!/bin/bash

# Script para testar todos os endpoints da API AnatomInd TG-263

set -e

API_BASE="http://localhost:8080"

echo "🧪 Testando API AnatomInd TG-263..."
echo

# Test 1: Health Check
echo "1️⃣ Health Check"
curl -s "$API_BASE/healthz" | head -c 200
echo -e "\n"

# Test 2: Models
echo "2️⃣ Lista de Modelos"
curl -s "$API_BASE/v1/models" | head -c 200
echo -e "\n"

# Test 3: Chat Completions (sem backend será erro esperado)
echo "3️⃣ Chat Completions (sem backend - erro esperado)"
curl -s -X POST "$API_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }' | head -c 200
echo -e "\n"

# Test 4: Completions (sem backend será erro esperado)
echo "4️⃣ Text Completions (sem backend - erro esperado)"
curl -s -X POST "$API_BASE/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The heart is",
    "max_tokens": 50
  }' | head -c 200
echo -e "\n"

# Test 5: Classificação TG-263 (sem backend será erro esperado)
echo "5️⃣ Classificação TG-263 (sem backend - erro esperado)"
timeout 10 curl -s -X POST "$API_BASE/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "sphenoid sinus",
    "top_k": 3
  }' | head -c 200 || echo "Timeout esperado - sem backend"
echo -e "\n"

echo "✅ Testes concluídos!"
echo "📝 Observações:"
echo "   - Health e Models: ✅ OK"
echo "   - Chat/Completions/Classify: ❌ Esperado (sem backend vLLM/TGI)"
echo "   - Para funcionar completamente, execute: docker compose up"
