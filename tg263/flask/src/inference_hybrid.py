#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência Híbrida TG-263: Fuzzy Matching + Modelo LLM
=====================================================

Sistema híbrido que combina:
1. Fuzzy matching inteligente (rápido) com tradução PT→EN
2. Modelo LLM com constrained decoding (preciso) quando confiança baixa

Estratégia:
- Primeiro: Tenta fuzzy matching com tradução médica
- Se confiança < limiar: Usa modelo LLM na GPU
- Retorna resultado com método usado e confiança
"""

import os
import csv
import json
import pandas as pd
import re
import torch
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Estruturas de suporte (Trie para constrained decoding)
# =============================================================================

class Trie:
    """Trie para constrained decoding"""
    def __init__(self):
        self.root: Dict[int, dict] = {}
        self.end = "_end_"

    def insert(self, token_ids: List[int]):
        node = self.root
        for tid in token_ids:
            node = node.setdefault(tid, {})
        node[self.end] = True

def build_label_trie(tokenizer, labels: List[str]) -> Trie:
    """Constrói trie para constrained decoding"""
    trie = Trie()
    for lab in labels:
        ids = tokenizer(" " + lab, add_special_tokens=False)["input_ids"]
        trie.insert(ids)
    return trie

def build_prefix_allowed_fn(tokenizer, trie: Trie, prefix_ids: List[int]):
    """Função para restringir tokens durante geração"""
    eos_id = tokenizer.eos_token_id

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        if isinstance(input_ids, torch.Tensor):
            seq = input_ids[batch_id].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            seq = input_ids[batch_id] if isinstance(input_ids, list) else input_ids
            if not isinstance(seq, list):
                seq = [seq]

        gen_path = seq[len(prefix_ids):] if len(seq) > len(prefix_ids) else []
        
        node = trie.root
        for tid in gen_path:
            if tid in node:
                node = node[tid]
            else:
                return [eos_id]

        allowed = [k for k in node.keys() if k != trie.end]
        if trie.end in node:
            allowed.append(eos_id)

        return allowed if allowed else [eos_id]

    return prefix_allowed_tokens_fn

# =============================================================================
# Classe Híbrida Principal
# =============================================================================

class TG263InferenceHybrid:
    def __init__(self, model_path: str, csv_path: str, fuzzy_threshold: float = 0.75):
        """
        Inferência híbrida: fuzzy matching + modelo LLM
        
        Args:
            model_path: Caminho para o modelo LLM
            csv_path: Caminho para o CSV com labels TG-263
            fuzzy_threshold: Limiar para usar LLM (se fuzzy_score < threshold)
        """
        logger.info("Inicializando TG263InferenceHybrid")
        
        self.fuzzy_threshold = fuzzy_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        logger.info(f"Device: {self.device}")
        
        # Carrega labels TG-263
        self._load_labels(csv_path)
        
        # Dicionário de tradução português → inglês
        self.pt_en_mapping = {
            'olho': 'eye', 'olhos': 'eyes',
            'mama': 'breast', 'mamas': 'breasts', 
            'cerebro': 'brain', 'coração': 'heart', 'coracao': 'heart',
            'figado': 'liver', 'fígado': 'liver',
            'rim': 'kidney', 'rins': 'kidneys',
            'pulmao': 'lung', 'pulmão': 'lung', 'pulmoes': 'lungs', 'pulmões': 'lungs',
            'estomago': 'stomach', 'estômago': 'stomach',
            'intestino': 'bowel', 'bexiga': 'bladder',
            'prostata': 'prostate', 'próstata': 'prostate',
            'utero': 'uterus', 'útero': 'uterus',
            'ovario': 'ovary', 'ovário': 'ovary', 'ovarios': 'ovaries', 'ovários': 'ovaries',
            'tireoide': 'thyroid', 'tireóide': 'thyroid',
            'esofago': 'esophagus', 'esôfago': 'esophagus',
            'reto': 'rectum', 'anus': 'anus', 'ânus': 'anus',
            'medula': 'spinal_cord', 'coluna': 'spine',
            'femur': 'femur', 'tibia': 'tibia', 'fibula': 'fibula',
            'humero': 'humerus', 'úmero': 'humerus',
            'radius': 'radius', 'rádio': 'radius', 'ulna': 'ulna',
            'clavicula': 'clavicle', 'clavícula': 'clavicle',
            'costela': 'rib', 'costelas': 'ribs',
            'vertebra': 'vertebra', 'vértebra': 'vertebra',
            'cranio': 'skull', 'crânio': 'skull',
            'mandibula': 'mandible', 'mandíbula': 'mandible',
            'maxila': 'maxilla',
            'dente': 'tooth', 'dentes': 'teeth',
            'lingua': 'tongue', 'língua': 'tongue',
            'laringe': 'larynx', 'faringe': 'pharynx',
            'traqueia': 'trachea', 'tráqueia': 'trachea'
        }
        
        # Inicializa apenas fuzzy matching por enquanto
        self.model = None
        self.tokenizer = None
        self.trie = None
        self.llm_available = False
        self.llm_load_attempted = False
        
        logger.info("TG263InferenceHybrid inicializado com fuzzy matching. LLM será carregado sob demanda.")

    def _load_labels(self, csv_path: str):
        """Carrega labels TG-263 do CSV"""
        logger.info(f"Carregando labels TG-263 de: {csv_path}")
        try:
            # Tenta diferentes separadores
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                    if 'TG263-Primary Name' in df.columns:
                        break
                except:
                    continue
            else:
                raise ValueError("Não foi possível ler o CSV")
            
            raw_labels = df['TG263-Primary Name'].dropna()
            self.tg_labels = []
            
            for label in raw_labels:
                try:
                    label_str = str(label).strip()
                    if label_str and label_str != 'nan' and len(label_str) > 0:
                        self.tg_labels.append(label_str)
                except:
                    continue
                    
            logger.info(f"Labels TG-263 carregadas: {len(self.tg_labels)}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            self.tg_labels = []

    def _load_llm_model(self, model_path: str):
        """Carrega modelo LLM para fallback sem timeout (fixado para Flask)"""
        logger.info(f"Tentando carregar modelo LLM de: /workspace/model")
        
        try:
            # Carrega tokenizer
            logger.info("Inicializando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Configura pad token se necessário
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Carrega modelo base
            logger.info("Carregando modelo base...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Move para device
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            # Constrói trie para constrained decoding
            logger.info("Construindo trie para constrained decoding...")
            self.trie = build_label_trie(self.tokenizer, self.tg_labels)
            
            logger.info(f"Modelo LLM carregado com sucesso em {self.device}")
            self.llm_available = True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo LLM: {e}")
            logger.warning("Continuando apenas com fuzzy matching")
            self.model = None
            self.tokenizer = None
            self.trie = None
            self.llm_available = False

    def normalize_medical_term(self, text: str) -> str:
        """Normaliza termo médico com tradução PT→EN e lateralidade"""
        if not text:
            return ""
        
        # Normalização básica
        text = text.strip().lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
        text = re.sub(r'\s+', ' ', text).strip()  # Normaliza espaços
        
        # Mapeamento de lateralidade português → inglês
        text = re.sub(r'\b(esquerdo|esquerda|esq|e)\b', 'L', text)
        text = re.sub(r'\b(direito|direita|dir|d)\b', 'R', text)
        
        # Tradução de termos médicos
        words = text.split()
        translated_words = []
        
        for word in words:
            if word in self.pt_en_mapping:
                translated_words.append(self.pt_en_mapping[word])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)

    def fuzzy_match(self, text: str) -> Dict:
        """Executa fuzzy matching com tradução médica"""
        if not self.tg_labels:
            return {
                "output": "NOT FOUND",
                "confidence": 0.0,
                "method": "no_labels",
                "rejected": True
            }
        
        # Normaliza e traduz o input
        normalized_input = self.normalize_medical_term(text)
        
        # Busca fuzzy no conjunto de labels
        matches = process.extract(
            normalized_input, 
            self.tg_labels, 
            scorer=fuzz.WRatio,
            limit=5
        )
        
        if not matches:
            return {
                "output": "NOT FOUND",
                "confidence": 0.0,
                "method": "fuzzy_no_match",
                "rejected": True
            }
        
        # Debug: verificar estrutura do match
        logger.info(f"Matches retornados: {matches[:2]}")  # Log primeiros 2 matches
        
        try:
            # process.extract retorna (match, score, index) - 3 elementos!
            best_match, score, _ = matches[0]  # Ignora o índice
            confidence = score / 100.0  # Normaliza para 0-1
        except ValueError as e:
            logger.error(f"Erro no unpacking de matches[0]: {matches[0]}, tipo: {type(matches[0])}")
            # Fallback: tentar extrair diretamente
            if isinstance(matches[0], tuple) and len(matches[0]) >= 2:
                best_match, score = matches[0][0], matches[0][1]
                confidence = score / 100.0
            else:
                logger.error(f"Formato de match inesperado: {matches[0]}")
                return {
                    "output": "FUZZY_ERROR", 
                    "confidence": 0.0,
                    "method": "fuzzy_error",
                    "rejected": True,
                    "error": str(e)
                }
        
        return {
            "output": best_match,
            "confidence": confidence,
            "method": "fuzzy_match",
            "rejected": False,
            "candidate_used": normalized_input,
            "score": score
        }

    def llm_inference(self, text: str) -> Dict:
        """Executa inferência com modelo LLM"""
        if not self.llm_available:
            return {
                "output": "MODEL NOT AVAILABLE",
                "confidence": 0.0,
                "method": "llm_unavailable", 
                "rejected": True
            }
        
        try:
            # Template de prompt
            prompt = f"Input: {text.strip()}\nOutput: "
            
            # Tokeniza prompt
            enc = self.tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            prefix_ids = enc["input_ids"][0].tolist()
            
            # Função para constrained decoding
            prefix_allowed_tokens_fn = build_prefix_allowed_fn(
                self.tokenizer, self.trie, prefix_ids
            )
            
            # Geração com constrained decoding
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=16,
                    do_sample=False,  # Greedy
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extrai tokens gerados
            seq = out.sequences[0]
            gen_ids = seq.tolist()[len(prefix_ids):]
            
            # Remove EOS se presente
            if gen_ids and gen_ids[-1] == self.tokenizer.eos_token_id:
                gen_ids = gen_ids[:-1]
            
            # Decodifica resultado
            pred_str = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_str = pred_str.split("\n")[0].strip()
            
            # Calcula confiança (log-prob média)
            if len(gen_ids) == 0:
                avg_logprob = float("-inf")
            else:
                step_scores = out.scores[:len(gen_ids)]
                logprobs = []
                for t, logits in enumerate(step_scores):
                    probs = torch.log_softmax(logits[0], dim=-1)
                    chosen_id = gen_ids[t]
                    logprobs.append(probs[chosen_id].item())
                avg_logprob = sum(logprobs) / max(1, len(logprobs))
            
            # Converte log-prob para confiança 0-1
            confidence = max(0.0, min(1.0, (avg_logprob + 5.0) / 5.0))
            
            return {
                "output": pred_str,
                "confidence": confidence,
                "method": "llm_constrained",
                "rejected": False,
                "avg_logprob": avg_logprob
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência LLM: {e}")
            return {
                "output": "LLM ERROR",
                "confidence": 0.0,
                "method": "llm_error",
                "rejected": True,
                "error": str(e)
            }

    def _ensure_llm_loaded(self):
        """Carrega modelo LLM sob demanda se necessário"""
        if self.llm_available or self.llm_load_attempted:
            return
        
        self.llm_load_attempted = True
        logger.info("Tentando carregar modelo LLM sob demanda...")
        self._load_llm_model(self.model_path)

    def predict(self, text: str) -> Dict:
        """
        Predição híbrida: fuzzy primeiro, LLM se necessário
        
        Tags especiais:
        - "[FORCE_LLM]texto" -> força uso do modelo LLM
        - "[FUZZY_ONLY]texto" -> força uso apenas do fuzzy
        """
        if not text or not text.strip():
            return {
                "output": "NOT FOUND",
                "confidence": 0.0,
                "method": "empty_input",
                "rejected": True
            }
        
        # Verificar tags especiais
        force_llm = False
        fuzzy_only = False
        original_text = text
        
        if text.startswith("[FORCE_LLM]"):
            force_llm = True
            text = text[11:].strip()  # Remove a tag
            logger.info(f"🚀 FORÇA LLM detectada para: '{text}'")
        elif text.startswith("[FUZZY_ONLY]"):
            fuzzy_only = True
            text = text[12:].strip()  # Remove a tag
            logger.info(f"🔍 FUZZY_ONLY detectado para: '{text}'")
        
        # Se forçar apenas fuzzy
        if fuzzy_only:
            logger.info("Usando apenas fuzzy matching (forçado)")
            result = self.fuzzy_match(text)
            result["method"] = "fuzzy_forced"
            return result
        
        # Se forçar LLM, carrega e usa diretamente
        if force_llm:
            logger.info("Forçando uso do modelo LLM...")
            self._ensure_llm_loaded()
            
            if self.llm_available:
                llm_result = self.llm_inference(text)
                llm_result["method"] = "llm_forced"
                logger.info(f"LLM forçado retornou: {llm_result}")
                return llm_result
            else:
                logger.warning("LLM forçado mas não disponível, usando fuzzy")
                result = self.fuzzy_match(text)
                result["method"] = "fuzzy_fallback_from_forced_llm"
                return result
        
        # Fase 1: Tenta fuzzy matching normal
        logger.info(f"Tentando fuzzy matching para: '{text}'")
        fuzzy_result = self.fuzzy_match(text)
        
        # Se fuzzy matching tem alta confiança, usa ele
        if fuzzy_result["confidence"] >= self.fuzzy_threshold:
            logger.info(f"Fuzzy matching bem-sucedido (confiança: {fuzzy_result['confidence']:.3f})")
            return fuzzy_result
        
        # Fase 2: Fuzzy falhou, tenta modelo LLM
        logger.info(f"Fuzzy matching baixa confiança ({fuzzy_result['confidence']:.3f}), tentando LLM...")
        
        # Carrega LLM sob demanda
        self._ensure_llm_loaded()
        
        if self.llm_available:
            llm_result = self.llm_inference(text)
            
            # Compara resultados e escolhe o melhor
            if llm_result["confidence"] > fuzzy_result["confidence"]:
                logger.info(f"LLM escolhido (confiança: {llm_result['confidence']:.3f})")
                return llm_result
            else:
                logger.info(f"Fuzzy escolhido mesmo com baixa confiança")
                return fuzzy_result
        else:
            # LLM não disponível, retorna fuzzy mesmo com baixa confiança
            logger.warning("LLM não disponível, usando fuzzy com baixa confiança")
            return fuzzy_result

    def get_diagnostics(self) -> Dict:
        """Retorna informações de diagnóstico do sistema"""
        return {
            "fuzzy_threshold": self.fuzzy_threshold,
            "llm_available": self.llm_available,
            "device": str(self.device),
            "total_labels": len(self.tg_labels),
            "translation_terms": len(self.pt_en_mapping),
            "model_status": "HYBRID: Fuzzy + LLM" if self.llm_available else "FUZZY ONLY"
        }
