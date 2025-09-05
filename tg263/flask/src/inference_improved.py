"""
Versão melhorada com mapeamento médico português/inglês e lateralidade
"""

import os
import logging
import pandas as pd
import re
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class TG263InferenceImproved:
    def __init__(self, model_path: str, csv_path: str):
        """
        Versão melhorada com mapeamento médico e lateralidade
        """
        logger.info(f"Inicializando TG263InferenceImproved")
        
        # Carrega labels
        logger.info(f"Carregando labels TG-263 de: {csv_path}")
        try:
            self.labels_df = pd.read_csv(csv_path, sep=';')
            raw_labels = self.labels_df['TG263-Primary Name'].dropna()
            
            self.tg_labels = []
            for label in raw_labels:
                try:
                    label_str = str(label).strip()
                    if label_str and label_str != 'nan' and len(label_str) > 0:
                        self.tg_labels.append(label_str)
                except Exception:
                    continue
                    
            logger.info(f"Labels TG-263 carregadas: {len(self.tg_labels)}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            self.tg_labels = []
        
        # Dicionário de tradução português → inglês para termos médicos
        self.pt_en_mapping = {
            # Estruturas principais
            'olho': 'eye',
            'olhos': 'eyes', 
            'mama': 'breast',
            'mamas': 'breasts',
            'cerebro': 'brain',
            'coração': 'heart',
            'coracao': 'heart',
            'figado': 'liver',
            'fígado': 'liver',
            'rim': 'kidney',
            'rins': 'kidneys',
            'pulmao': 'lung',
            'pulmão': 'lung',
            'pulmoes': 'lungs',
            'pulmões': 'lungs',
            'estomago': 'stomach',
            'estômago': 'stomach',
            'bexiga': 'bladder',
            'reto': 'rectum',
            'prostata': 'prostate',
            'próstata': 'prostate',
            'tireoide': 'thyroid',
            'tireóide': 'thyroid',
            
            # Partes do olho
            'cornea': 'cornea',
            'córnea': 'cornea',
            'retina': 'retina',
            'lente': 'lens',
            'orbita': 'orbit',
            'órbita': 'orbit',
            
            # Outros
            'osso': 'bone',
            'musculo': 'muscle',
            'músculo': 'muscle',
            'veia': 'vein',
            'arteria': 'artery',
            'artéria': 'artery'
        }
        
        # Mapeamento de lateralidade
        self.laterality_mapping = {
            # Português
            'e': '_L',    # esquerdo
            'esq': '_L',  # esquerdo
            'esquerdo': '_L',
            'esquerda': '_L',
            'd': '_R',    # direito  
            'dir': '_R',  # direito
            'direito': '_R',
            'direita': '_R',
            'bilateral': 's',  # plurais (eyes, kidneys, etc)
            'ambos': 's',
            
            # Inglês
            'l': '_L',
            'left': '_L',
            'r': '_R', 
            'right': '_R',
            'both': 's'
        }
        
        logger.info("Dicionários de tradução e lateralidade carregados")
        logger.info("TG263InferenceImproved inicializado com sucesso!")

    def normalize_medical_term(self, text: str) -> list:
        """
        Normaliza termo médico português para possíveis variações em inglês
        """
        text = text.lower().strip()
        
        # Lista de possíveis traduções
        candidates = []
        
        # 1. Extrair lateralidade se presente
        laterality = None
        base_term = text
        
        # Buscar padrões de lateralidade no final
        for pt_lat, en_lat in self.laterality_mapping.items():
            if text.endswith(pt_lat.lower()):
                laterality = en_lat
                base_term = text[:-len(pt_lat)].strip()
                break
        
        # 2. Traduzir termo base
        translated_terms = []
        
        # Tradução direta
        if base_term in self.pt_en_mapping:
            translated_terms.append(self.pt_en_mapping[base_term])
        
        # Adicionar termo original caso seja inglês
        translated_terms.append(base_term)
        
        # Busca parcial por termos compostos
        for pt_term, en_term in self.pt_en_mapping.items():
            if pt_term in base_term or base_term in pt_term:
                translated_terms.append(en_term)
        
        # 3. Gerar candidatos finais
        for term in translated_terms:
            if laterality:
                if laterality == 's':
                    # Plural
                    if not term.endswith('s'):
                        candidates.append(term + 's')
                    candidates.append(term)
                else:
                    # Específico (L/R)
                    candidates.append(term + laterality)
                    candidates.append(term.capitalize() + laterality)
            
            # Sempre adicionar termo sem lateralidade
            candidates.append(term)
            candidates.append(term.capitalize())
        
        # Remover duplicatas mantendo ordem
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        
        return unique_candidates

    def predict(self, text: str) -> dict:
        """
        Predição melhorada com mapeamento médico
        """
        logger.info(f"Processando: '{text}'")
        
        if not text or not text.strip():
            return {
                "output": "NOT FOUND",
                "confidence": -float('inf'),
                "rejected": True,
                "method": "empty_input"
            }
        
        try:
            # 1. Gerar candidatos traduzidos
            candidates = self.normalize_medical_term(text)
            logger.info(f"Candidatos gerados: {candidates}")
            
            best_score = 0
            best_label = None
            best_method = "no_match"
            
            # 2. Buscar match exato primeiro
            for candidate in candidates:
                for label in self.tg_labels:
                    if candidate.lower() == label.lower():
                        logger.info(f"Match exato encontrado: {candidate} → {label}")
                        return {
                            "output": label,
                            "confidence": 1.0,
                            "rejected": False,
                            "method": "exact_match_translated",
                            "candidate_used": candidate
                        }
            
            # 3. Fuzzy matching nos candidatos traduzidos
            for candidate in candidates:
                for label in self.tg_labels:
                    try:
                        score = fuzz.ratio(candidate.lower(), label.lower())
                        
                        if score > best_score:
                            best_score = score
                            best_label = label
                            best_method = "fuzzy_translated"
                        
                    except Exception:
                        continue
            
            # 4. Fuzzy matching direto no termo original
            original_text = text.lower().strip()
            for label in self.tg_labels:
                try:
                    score = fuzz.ratio(original_text, label.lower())
                    
                    if score > best_score:
                        best_score = score
                        best_label = label
                        best_method = "fuzzy_direct"
                    
                except Exception:
                    continue
            
            # 5. Retornar resultado
            if best_label and best_score >= 70:  # Threshold mais alto para qualidade
                confidence = best_score / 100.0
                rejected = confidence < 0.8  # Threshold mais exigente
                
                logger.info(f"Melhor match: {best_label} (score: {best_score}, method: {best_method})")
                
                return {
                    "output": best_label,
                    "confidence": confidence,
                    "rejected": rejected,
                    "method": best_method,
                    "score": best_score,
                    "candidates": candidates
                }
            
            # 6. Nenhum match adequado
            logger.warning(f"Nenhum match adequado para: '{text}' (melhor score: {best_score})")
            return {
                "output": "NOT FOUND",
                "confidence": -float('inf'),
                "rejected": True,
                "method": "no_match",
                "best_score": best_score if best_score > 0 else None,
                "candidates": candidates
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
        Informações de diagnóstico melhoradas
        """
        return {
            "total_labels": len(self.tg_labels),
            "sample_labels": self.tg_labels[:10] if len(self.tg_labels) >= 10 else self.tg_labels,
            "model_status": "MEDICAL TRANSLATION + FUZZY MATCHING",
            "features": [
                "Portuguese ↔ English translation",
                "Laterality mapping (D/E → R/L)",
                "Medical terminology awareness",
                "Higher quality thresholds"
            ],
            "translation_terms": len(self.pt_en_mapping),
            "api_status": "IMPROVED"
        }
