"""
Versão de teste básica para isolamento de problemas
"""

import os
import logging
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class TG263InferenceBasic:
    def __init__(self, csv_path: str):
        """
        Versão básica sem modelo LLM - apenas fuzzy matching
        """
        logger.info(f"Inicializando TG263InferenceBasic (sem modelo)")
        
        # Carrega labels
        logger.info(f"Carregando labels TG-263 de: {csv_path}")
        self.labels_df = pd.read_csv(csv_path, sep=';')
        self.tg_labels = self.labels_df['TG263-Primary Name'].tolist()
        logger.info(f"Labels TG-263 carregadas: {len(self.tg_labels)}")
        
        logger.info("TG263InferenceBasic inicializado com sucesso!")

    def translate_basic(self, text: str) -> tuple[str, float]:
        """
        Traduz usando apenas fuzzy matching direto
        """
        try:
            logger.info(f"Fazendo fuzzy matching para: '{text}'")
            
            # Fuzzy matching direto
            best_match = process.extractOne(
                text, 
                self.tg_labels, 
                scorer=fuzz.ratio
            )
            
            if best_match and best_match[1] >= 70:  # Score mínimo de 70%
                confidence = best_match[1] / 100.0
                logger.info(f"Melhor match: {best_match[0]} (score: {best_match[1]})")
                return best_match[0], confidence
            else:
                logger.warning(f"Nenhum match adequado encontrado para: '{text}'")
                return "NOT FOUND", -float('inf')
            
        except Exception as e:
            logger.error(f"Erro durante fuzzy matching: {e}")
            return "NOT FOUND", -float('inf')

    def predict(self, text: str) -> dict:
        """
        Faz predição completa
        """
        logger.info(f"Processando: '{text}'")
        
        output, confidence = self.translate_basic(text)
        
        # Determina se foi rejeitado
        rejected = confidence < 0.7  # 70%
        
        result = {
            "output": output,
            "confidence": confidence,
            "rejected": rejected
        }
        
        logger.info(f"Resultado básico: {result}")
        return result
