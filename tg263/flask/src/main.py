#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Flask para inferência TG-263
=================================

API simples para traduzir nomes de estruturas anatômicas para o padrão TG-263.
"""

import os
import logging
from flask import Flask, request, jsonify
from inference_hybrid import TG263InferenceHybrid

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Criar app Flask
app = Flask(__name__)

# Configurações via variáveis de ambiente
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model/Llama-8B")
TG_CSV_PATH = os.getenv("TG_CSV_PATH", "/workspace/labels/TG263.csv")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "-2.5"))
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Instância global do modelo (carregada na inicialização)
inference_model = None

def initialize_model():
    """Inicializa o modelo de inferência."""
    global inference_model
    
    try:
        logger.info("Inicializando modelo TG-263...")
        inference_model = TG263InferenceHybrid(
            model_path=MODEL_PATH,
            csv_path=TG_CSV_PATH,
            fuzzy_threshold=0.75  # Se fuzzy < 75%, usa LLM
        )
        logger.info("Modelo inicializado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar modelo: {e}")
        return False

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint de health check."""
    if inference_model is None:
        return jsonify({"status": "error", "message": "Modelo não carregado"}), 503
    
    return jsonify({
        "status": "healthy",
        "model_path": MODEL_PATH,
        "tg_csv_path": TG_CSV_PATH
    })

@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    """
    Endpoint para diagnóstico do sistema
    """
    if inference_model is None:
        return jsonify({"error": "Modelo não carregado"}), 503
    
    try:
        diag_info = inference_model.get_diagnostics()
        return jsonify(diag_info)
    except Exception as e:
        logger.error(f"Erro no diagnóstico: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal para predição.
    
    Esperado JSON:
    {
        "text": "estrutura anatômica",
        "min_confidence": -2.5  // opcional
    }
    
    Retorna:
    {
        "output": "TG_263_Label",
        "confidence": -1.2,
        "rejected": false
    }
    """
    if inference_model is None:
        return jsonify({"error": "Modelo não carregado"}), 503
    
    try:
        # Obter dados do request
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON inválido"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Campo 'text' é obrigatório"}), 400
        
        min_conf = data.get("min_confidence", MIN_CONFIDENCE)
        
        logger.info(f"Predição solicitada: '{text[:50]}...' (confiança mín: {min_conf})")
        
        # Fazer predição
        result = inference_model.predict(text)
        
        # Aplicar filtro de confiança mínima
        if result['confidence'] < min_conf:
            result['rejected'] = True
        
        logger.info(f"Resultado: {result['output']} (conf: {result['confidence']:.3f}, rejeitado: {result['rejected']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Endpoint para predição em lote.
    
    Esperado JSON:
    {
        "texts": ["estrutura1", "estrutura2", ...],
        "min_confidence": -2.5  // opcional
    }
    
    Retorna:
    {
        "results": [
            {"output": "TG_Label1", "confidence": -1.2, "rejected": false},
            {"output": "NOT FOUND", "confidence": -3.0, "rejected": true},
            ...
        ]
    }
    """
    if inference_model is None:
        return jsonify({"error": "Modelo não carregado"}), 503
    
    try:
        # Obter dados do request
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON inválido"}), 400
        
        texts = data.get("texts", [])
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Campo 'texts' deve ser uma lista não-vazia"}), 400
        
        if len(texts) > 100:  # Limitar tamanho do batch
            return jsonify({"error": "Máximo 100 textos por batch"}), 400
        
        min_conf = data.get("min_confidence", MIN_CONFIDENCE)
        
        # Fazer predições
        results = inference_model.predict_batch(texts, min_confidence=min_conf)
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/labels", methods=["GET"])
def get_labels():
    """
    Endpoint para obter todas as labels TG-263 disponíveis.
    """
    if inference_model is None:
        return jsonify({"error": "Modelo não carregado"}), 503
    
    try:
        return jsonify({
            "labels": inference_model.tg_labels,
            "count": len(inference_model.tg_labels)
        })
    except Exception as e:
        logger.error(f"Erro ao obter labels: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    """Endpoint raiz com informações da API."""
    return jsonify({
        "name": "TG-263 Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predição single (POST)",
            "/predict_batch": "Predição em lote (POST)",
            "/labels": "Obter todas as labels TG-263"
        },
        "model_loaded": inference_model is not None
    })

if __name__ == "__main__":
    logger.info("Iniciando API Flask TG-263...")
    logger.info(f"Configurações:")
    logger.info(f"  MODEL_PATH: {MODEL_PATH}")
    logger.info(f"  TG_CSV_PATH: {TG_CSV_PATH}")
    logger.info(f"  MIN_CONFIDENCE: {MIN_CONFIDENCE}")
    logger.info(f"  HOST: {HOST}")
    logger.info(f"  PORT: {PORT}")
    
    # Inicializar modelo
    if not initialize_model():
        logger.error("Falha ao inicializar modelo. Saindo...")
        exit(1)
    
    # Iniciar servidor
    logger.info(f"Iniciando servidor em {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
