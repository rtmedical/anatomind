"""
Versão final robusta com fallback completo para fuzzy matching
Documenta os problemas encontrados com o modelo Llama-8B
"""

import os
import logging
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class TG263InferenceFinal:
    def __init__(self, model_path: str, csv_path: str):
        """
        Versão final robusta que documenta problemas encontrados
        """
        logger.info(f"Inicializando TG263InferenceFinal")
        
        # Carrega labels
        logger.info(f"Carregando labels TG-263 de: {csv_path}")
        self.labels_df = pd.read_csv(csv_path, sep=';')
        # Filtrar apenas valores string válidos
        raw_labels = self.labels_df['TG263-Primary Name'].dropna()
        self.tg_labels = [str(label).strip() for label in raw_labels if str(label).strip() and str(label) != 'nan']
        logger.info(f"Labels TG-263 carregadas: {len(self.tg_labels)}")
        
        # Log do problema identificado
        logger.warning("="*60)
        logger.warning("PROBLEMA IDENTIFICADO COM MODELO LLAMA-8B:")
        logger.warning("- Erro CUDA 'device-side assert triggered'")
        logger.warning("- Index out of bounds em IndexKernel.cu:92")
        logger.warning("- Modelo gera tokens fora do vocabulário válido")
        logger.warning("- Problema persiste mesmo com configurações conservadoras")
        logger.warning("- Solução: usar fuzzy matching como método principal")
        logger.warning("="*60)
        
        logger.info("TG263InferenceFinal inicializado - usando fuzzy matching otimizado")

    def predict(self, text: str) -> dict:
        """
        Predição usando fuzzy matching otimizado
        """
        logger.info(f"Processando: '{text}'")
        
        try:
            # Estratégia multi-nível de fuzzy matching
            best_matches = []
            
            # 1. Match exato (case insensitive)
            for label in self.tg_labels:
                if text.lower().strip() == label.lower().strip():
                    logger.info(f"Match exato encontrado: {label}")
                    return {
                        "output": label,
                        "confidence": 1.0,
                        "rejected": False,
                        "method": "exact_match"
                    }
            
            # 2. Match por palavra-chave principal
            text_words = text.lower().split()
            for word in text_words:
                if len(word) >= 3:  # Palavras com pelo menos 3 caracteres
                    for label in self.tg_labels:
                        if word in label.lower():
                            score = fuzz.ratio(text.lower(), label.lower())
                            best_matches.append((label, score, "keyword"))
            
            # 3. Fuzzy matching tradicional
            fuzzy_results = process.extract(
                text, 
                self.tg_labels, 
                scorer=fuzz.ratio,
                limit=5
            )
            
            for label, score in fuzzy_results:
                best_matches.append((label, score, "fuzzy"))
            
            # 4. Fuzzy matching com partial ratio (para textos compostos)
            partial_results = process.extract(
                text,
                self.tg_labels,
                scorer=fuzz.partial_ratio,
                limit=3
            )
            
            for label, score in partial_results:
                # Partial ratio tem peso menor
                adjusted_score = score * 0.8
                best_matches.append((label, adjusted_score, "partial"))
            
            # Encontrar melhor match geral
            if best_matches:
                # Ordenar por score
                best_matches.sort(key=lambda x: x[1], reverse=True)
                best_label, best_score, method = best_matches[0]
                
                confidence = best_score / 100.0
                rejected = confidence < 0.6  # Threshold mais relaxado
                
                result = {
                    "output": best_label,
                    "confidence": confidence,
                    "rejected": rejected,
                    "method": method,
                    "candidates": [
                        {"label": label, "score": score, "method": method}
                        for label, score, method in best_matches[:3]
                    ]
                }
                
                logger.info(f"Melhor match: {best_label} (score: {best_score:.1f}, method: {method})")
                return result
            
            # Nenhum match encontrado
            logger.warning(f"Nenhum match encontrado para: '{text}'")
            return {
                "output": "NOT FOUND",
                "confidence": -float('inf'),
                "rejected": True,
                "method": "no_match",
                "candidates": []
            }
            
        except Exception as e:
            logger.error(f"Erro durante processamento: {e}")
            return {
                "output": "NOT FOUND",
                "confidence": -float('inf'),
                "rejected": True,
                "method": "error",
                "error": str(e)
            }

    def get_diagnostics(self) -> dict:
        """
        Retorna informações de diagnóstico do sistema
        """
        return {
            "total_labels": len(self.tg_labels),
            "sample_labels": self.tg_labels[:10],
            "model_status": "DISABLED - CUDA errors with Llama-8B",
            "primary_method": "fuzzy_matching",
            "cuda_issue": {
                "error": "device-side assert triggered",
                "location": "IndexKernel.cu:92", 
                "cause": "tokens out of vocabulary bounds",
                "solution": "using fuzzy matching instead"
            }
        }
