#!/bin/bash
set -e

echo "=== AnatomInd API - Exemplos de Uso ==="
echo

BASE_URL="http://localhost:8080"

echo "1. Health Check"
echo "GET $BASE_URL/healthz"
curl -s "$BASE_URL/healthz" | jq '.'
echo
echo

echo "2. Listar Modelos"
echo "GET $BASE_URL/v1/models"
curl -s "$BASE_URL/v1/models" | jq '.'
echo
echo

echo "3. Chat Completion (não-streaming, provider padrão)"
echo "POST $BASE_URL/v1/chat/completions"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the lung structure in TG-263?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }' | jq '.'
echo
echo

echo "4. Chat Completion (não-streaming, provider específico)"
echo "POST $BASE_URL/v1/chat/completions?provider=vllm"
curl -s "$BASE_URL/v1/chat/completions?provider=vllm" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Describe the heart structure"}
    ],
    "temperature": 0.5,
    "max_tokens": 30
  }' | jq '.'
echo
echo

echo "5. Chat Completion (streaming)"
echo "POST $BASE_URL/v1/chat/completions (stream=true)"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about brain anatomy"}
    ],
    "stream": true,
    "max_tokens": 20
  }'
echo
echo

echo "6. Completions (não-streaming)"
echo "POST $BASE_URL/v1/completions"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The spleen is located in",
    "temperature": 0.3,
    "max_tokens": 25
  }' | jq '.'
echo
echo

echo "7. Classificação TG-263 (provider padrão)"
echo "POST $BASE_URL/classify"
curl -s "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "sphenoid sinus structure",
    "top_k": 3
  }' | jq '.'
echo
echo

echo "8. Classificação TG-263 (provider específico)"
echo "POST $BASE_URL/classify"
curl -s "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "lung parenchyma",
    "top_k": 5,
    "provider": "tgi"
  }' | jq '.'
echo
echo

echo "9. Classificação TG-263 (teste threshold)"
echo "POST $BASE_URL/classify"
curl -s "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "random unrelated text that should not match",
    "top_k": 2
  }' | jq '.'
echo
echo

echo "=== Fim dos Exemplos ==="
