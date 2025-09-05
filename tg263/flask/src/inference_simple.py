#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versão simplificada sem constrained decoding
"""

import os
import csv
import json
import pandas as pd
from typing import List, Tuple, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rapidfuzz import process, fuzz
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_label_set(path_csv: str, column: str = "TG263-Primary Name") -> List[str]:
    """
    Carrega as labels TG-263 do arquivo CSV.
    """
    logger.info(f"Carregando labels TG-263 de: {path_csv}")
    
    ext = os.path.splitext(path_csv)[1].lower()
    
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path_csv)
    else:
        # Tentar diferentes separadores
        try:
            df = pd.read_csv(path_csv, sep=';', encoding='utf-8-sig')
        except Exception:
            try:
                df = pd.read_csv(path_csv, sep=',', encoding='utf-8-sig')
            except Exception:
                df = pd.read_csv(path_csv, encoding='utf-8-sig')
    
    if df is None or df.empty:
        raise ValueError(f"DataFrame vazio após leitura de {path_csv}")

    # Sanitizar nomes de colunas
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    
    # Buscar coluna target
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("-", "")
    
    wanted_norm = _norm(column)
    candidates = {_norm(c): c for c in df.columns}

    tg_col = None
    if wanted_norm in candidates:
        tg_col = candidates[wanted_norm]
    else:
        # Busca heurística
        for c in df.columns:
            if 'tg263' in c.lower() and ('primary' in c.lower() or 'name' in c.lower()):
                tg_col = c
                break
    
    if tg_col is None:
        raise ValueError(f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    # Extrair labels válidas
    raw_labels = df[tg_col].dropna().astype(str).tolist()
    
    valid_labels = []
    for label in raw_labels:
        clean_label = label.replace("\ufeff", "").strip()
        if clean_label and len(clean_label) < 100:
            valid_labels.append(clean_label)
    
    unique_labels = sorted(list(set(valid_labels)), key=len)
    
    if len(unique_labels) == 0:
        raise ValueError("Nenhuma label válida encontrada!")
    
    logger.info(f"Labels TG-263 carregadas: {len(unique_labels)}")
    return unique_labels


def translate_simple(
    model,
    tokenizer,
    text_in: str,
    device,
    max_new_tokens: int = 16,
    reject_if_low_conf: bool = True,
    min_avg_logprob: float = -2.5,
) -> Tuple[Optional[str], Dict]:
    """
    Gera um rótulo sem constrained decoding.
    """
    
    def _normalize_inp(s: str) -> str:
        return " ".join(s.split())  # Normalizar espaços
    
    prompt = f"Input: {_normalize_inp(text_in)}\nOutput: "
    
    try:
        # Tokenização do prompt
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        prefix_ids = enc["input_ids"][0].tolist()

        with torch.no_grad():
            # Limpar cache antes da geração
            if device == "cuda":
                torch.cuda.empty_cache()
                
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = out.sequences[0]
        
        # IDs realmente gerados
        gen_ids = seq.tolist()[len(prefix_ids):]
        if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
            gen_ids = gen_ids[:-1]

        # Decodificar apenas a parte gerada
        pred_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_str = pred_str.split("\n")[0].strip()

        # Calcular confiança
        if len(gen_ids) == 0:
            avg_logprob = float("-inf")
        else:
            scores = torch.stack(out.scores[:len(gen_ids)], dim=0)
            logprobs = torch.log_softmax(scores, dim=-1)
            selected_logprobs = logprobs[range(len(gen_ids)), gen_ids]
            avg_logprob = selected_logprobs.mean().item()

        meta = {"avg_logprob": avg_logprob, "rejected": False}

        if reject_if_low_conf and avg_logprob < min_avg_logprob:
            meta["rejected"] = True
            return None, meta

        return pred_str, meta
        
    except Exception as e:
        logger.error(f"Erro durante geração: {e}")
        meta = {"avg_logprob": float("-inf"), "rejected": True, "error": str(e)}
        return None, meta


class TG263Inference:
    """
    Classe principal para inferência do modelo TG-263 (versão simplificada).
    """
    
    def __init__(self, model_path: str, tg_csv_path: str, device: str = "auto"):
        self.model_path = model_path
        self.tg_csv_path = tg_csv_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Inicializando TG263Inference no device: {self.device}")
        
        # Carregar labels TG-263
        self.tg_labels = load_label_set(tg_csv_path)
        
        # Carregar modelo e tokenizer
        self._load_model()
        
        logger.info("TG263Inference inicializado com sucesso!")
    
    def _load_model(self):
        """Carrega o modelo e tokenizer."""
        logger.info(f"Carregando modelo de: {self.model_path}")
        
        try:
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Garantir pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configurar carregamento do modelo com otimizações de memória
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if self.device == "cuda":
                # Configurações para CUDA
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "max_memory": {0: "20GB"},  # Limitar uso de memória da GPU
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                })
            
            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Configurações para otimização
            self.model.eval()
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                
            # Limpar cache da GPU se necessário
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            logger.info("Modelo carregado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def predict(self, text_in: str, min_confidence: float = -2.5) -> Dict:
        """
        Prediz o rótulo TG-263 para um texto de entrada.
        
        Args:
            text_in: Texto da estrutura anatômica
            min_confidence: Limiar mínimo de confiança
            
        Returns:
            Dict com 'output', 'confidence', 'rejected'
        """
        if not text_in or not text_in.strip():
            return {
                "output": "NOT FOUND",
                "confidence": float("-inf"),
                "rejected": True,
                "error": "Input vazio"
            }
        
        try:
            # Usar geração simples sem constrained decoding
            pred, meta = translate_simple(
                self.model,
                self.tokenizer,
                text_in.strip(),
                self.device,
                reject_if_low_conf=True,
                min_avg_logprob=min_confidence
            )
            
            if pred is None or meta["rejected"]:
                return {
                    "output": "NOT FOUND",
                    "confidence": meta["avg_logprob"],
                    "rejected": True
                }
            
            # Pós-processamento: verificar se a predição é uma label válida
            if pred in self.tg_labels:
                return {
                    "output": pred,
                    "confidence": meta["avg_logprob"],
                    "rejected": False
                }
            
            # Se não for uma label exata, tentar encontrar a mais similar
            match, score, _ = process.extractOne(pred, self.tg_labels, scorer=fuzz.ratio)
            
            if score >= 80:  # Limiar de similaridade
                logger.info(f"Mapeamento fuzzy: '{pred}' -> '{match}' (score: {score})")
                return {
                    "output": match,
                    "confidence": meta["avg_logprob"] * (score / 100),  # Penalizar confiança
                    "rejected": False
                }
            
            return {
                "output": "NOT FOUND",
                "confidence": meta["avg_logprob"],
                "rejected": True
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                "output": "NOT FOUND",
                "confidence": float("-inf"),
                "rejected": True,
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str], min_confidence: float = -2.5) -> List[Dict]:
        """
        Prediz rótulos TG-263 para uma lista de textos.
        """
        results = []
        for text in texts:
            result = self.predict(text, min_confidence)
            results.append(result)
        return results
