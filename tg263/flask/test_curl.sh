#!/bin/bash
# Script de teste da API Flask TG-263 usando curl

API_URL="http://localhost:5000"

echo "==============================================="
echo "TESTE DA API FLASK TG-263 COM CURL"
echo "==============================================="

echo
echo "🏠 Testando endpoint raiz..."
curl -s "$API_URL/" | jq .

echo
echo "🔍 Testando health check..."
curl -s "$API_URL/health" | jq .

echo
echo "🧠 Testando predição single - Kidney_L..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Kidney_L"}' | jq .

echo
echo "🧠 Testando predição single - rim esquerdo..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "rim esquerdo"}' | jq .

echo
echo "🧠 Testando predição single - coração..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "coração"}' | jq .

echo
echo "📦 Testando predição em lote..."
curl -s -X POST "$API_URL/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Kidney_L", "Heart", "Liver", "estrutura inexistente"]}' | jq .

echo
echo "📋 Testando obtenção das labels (primeiras 10)..."
curl -s "$API_URL/labels" | jq '.labels[:10]'

echo
echo "✅ Testes concluídos!"
