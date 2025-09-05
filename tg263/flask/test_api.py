#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliente de teste para a API Flask TG-263
========================================

Script para testar os endpoints da API.
"""

import requests
import json

# Configuração da API
API_URL = "http://localhost:5000"

def test_health():
    """Testa o endpoint de health check."""
    print("🔍 Testando health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {response.json()}")
    print()

def test_predict_single():
    """Testa predição de um único texto."""
    print("🧠 Testando predição single...")
    
    test_cases = [
        "Kidney_L",
        "rim esquerdo", 
        "coração",
        "fígado",
        "estrutura inexistente xyz123"
    ]
    
    for text in test_cases:
        data = {"text": text}
        response = requests.post(f"{API_URL}/predict", json=data)
        print(f"Input: '{text}'")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Output: {result['output']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Rejected: {result['rejected']}")
        else:
            print(f"Erro: {response.text}")
        print("-" * 40)

def test_predict_batch():
    """Testa predição em lote."""
    print("📦 Testando predição em lote...")
    
    texts = ["Kidney_L", "Heart", "Liver", "Brain", "estrutura inexistente"]
    data = {"texts": texts}
    
    response = requests.post(f"{API_URL}/predict_batch", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()["results"]
        for i, result in enumerate(results):
            print(f"Input {i+1}: '{texts[i]}'")
            print(f"  Output: {result['output']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Rejected: {result['rejected']}")
    else:
        print(f"Erro: {response.text}")
    print()

def test_get_labels():
    """Testa obtenção das labels."""
    print("📋 Testando obtenção das labels...")
    response = requests.get(f"{API_URL}/labels")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total de labels: {data['count']}")
        print(f"Primeiras 10 labels: {data['labels'][:10]}")
    else:
        print(f"Erro: {response.text}")
    print()

def test_root():
    """Testa endpoint raiz."""
    print("🏠 Testando endpoint raiz...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DA API FLASK TG-263")
    print("=" * 60)
    
    try:
        test_root()
        test_health()
        test_predict_single()
        test_predict_batch()
        test_get_labels()
        
        print("✅ Todos os testes concluídos!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Erro: Não foi possível conectar à API. Certifique-se de que ela esteja rodando.")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
