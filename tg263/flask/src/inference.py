#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de inferência para o modelo TG-263
==========================================

Este módulo implementa as funções de inferência para traduzir nomes de estruturas
anatômicas para o padrão TG-263, usando constrained decoding via trie.

Baseado no script de treinamento original, mas otimizado para uso em produção.
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

# =============================================================================
# Estruturas de suporte (Trie para constrained decoding)
# =============================================================================

class Trie:
    """
    Trie mínima para restringir tokens. Cada caminho completo termina em `self.end`.
    """
    def __init__(self):
        self.root: Dict[int, dict] = {}
        self.end = "_end_"

    def insert(self, token_ids: List[int]):
        node = self.root
        for tid in token_ids:
            node = node.setdefault(tid, {})
        node[self.end] = True


def build_label_trie(tokenizer, labels: List[str]) -> Trie:
    """
    Constrói o trie com as labels TG-263 tokenizadas **como na geração**.
    Importante: usamos `" " + label` para casar com o espaço após "Output: ".
    """
    trie = Trie()
    vocab_size = tokenizer.vocab_size
    
    for lab in labels:
        ids = tokenizer(" " + lab, add_special_tokens=False)["input_ids"]
        
        # Validar todos os tokens antes de inserir no trie
        valid_ids = []
        for token_id in ids:
            if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                valid_ids.append(token_id)
            else:
                logger.warning(f"Token inválido ignorado para label '{lab}': {token_id}")
        
        if valid_ids:  # Só inserir se há tokens válidos
            trie.insert(valid_ids)
        else:
            logger.warning(f"Label '{lab}' não inserida no trie - nenhum token válido")
    
    return trie


# =============================================================================
# Carregamento de labels TG-263
# =============================================================================

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


# =============================================================================
# Constrained decoding
# =============================================================================

def build_prefix_allowed_fn(tokenizer, trie: Trie, prefix_ids: List[int]):
    """
    Retorna função `prefix_allowed_tokens_fn` para HF generate(), restringindo os
    próximos tokens aos caminhos **válidos** do trie (labels TG-263).
    """
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    vocab_size = tokenizer.vocab_size

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        try:
            # Extrair parte gerada (após o prefixo)
            if len(input_ids) <= len(prefix_ids):
                # No início da geração, permitir tokens da raiz do trie + EOS
                root_tokens = []
                for key in trie.root.keys():
                    if isinstance(key, int) and 0 <= key < vocab_size:
                        root_tokens.append(key)
                
                # Sempre adicionar EOS como opção
                if eos_id not in root_tokens:
                    root_tokens.append(eos_id)
                
                # Se raiz vazia, permitir pelo menos alguns tokens comuns
                if not root_tokens:
                    logger.warning("Raiz do trie vazia - usando tokens padrão")
                    return [eos_id, 220, 32]  # EOS, espaço, alguns tokens comuns
                
                return root_tokens
            
            gen_ids = input_ids[len(prefix_ids):].tolist()
            
            # Navegar no trie
            node = trie.root
            for tid in gen_ids:
                if tid in node:
                    node = node[tid]
                else:
                    # Caminho inválido - permitir apenas EOS para terminar
                    return [eos_id]
            
            # Tokens permitidos a partir deste nó
            allowed = []
            for key in node.keys():
                # Filtrar apenas tokens válidos (inteiros e dentro do vocabulário)
                if key != trie.end and isinstance(key, int) and 0 <= key < vocab_size:
                    allowed.append(key)
            
            # Se fim de palavra permitido, adicionar EOS
            if trie.end in node:
                allowed.append(eos_id)
            
            # Garantir que nunca retornamos lista vazia
            if not allowed:
                logger.warning(f"Nenhum token permitido no nó atual - usando EOS. Gen_ids: {gen_ids}")
                allowed = [eos_id]
            
            return allowed
            
        except Exception as e:
            logger.error(f"Erro em prefix_allowed_tokens_fn: {e}")
            # Em caso de erro, sempre retornar pelo menos EOS
            return [eos_id]

    return prefix_allowed_tokens_fn


def translate_constrained(
    model,
    tokenizer,
    trie: Trie,
    text_in: str,
    device,
    max_new_tokens: int = 16,
    reject_if_low_conf: bool = True,
    min_avg_logprob: float = -2.5,
) -> Tuple[Optional[str], Dict]:
    """
    Gera **um** rótulo TG-263 com decoding restrito ao trie e calcula a
    log-prob média por token gerado.
    """
    
    def _normalize_inp(s: str) -> str:
        return " ".join(s.split())  # Normalizar espaços
    
    prompt = f"Input: {_normalize_inp(text_in)}\nOutput: "
    
    try:
        # Tokenização do prompt
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        prefix_ids = enc["input_ids"][0].tolist()

        prefix_allowed_tokens_fn = build_prefix_allowed_fn(tokenizer, trie, prefix_ids)

        with torch.no_grad():
            # Limpar cache antes da geração para evitar problemas de memória
            if device == "cuda":
                torch.cuda.empty_cache()
                
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM durante geração: {e}")
        # Limpar cache e tentar novamente com configurações mais conservadoras
        torch.cuda.empty_cache()
        meta = {"avg_logprob": float("-inf"), "rejected": True, "error": "CUDA OOM"}
        return None, meta
        
    except Exception as e:
        logger.error(f"Erro durante geração: {e}")
        meta = {"avg_logprob": float("-inf"), "rejected": True, "error": str(e)}
        return None, meta


# =============================================================================
# Classe principal de inferência
# =============================================================================

class TG263Inference:
    """
    Classe principal para inferência do modelo TG-263.
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
        
        # Construir trie
        self.trie = build_label_trie(self.tokenizer, self.tg_labels)
        
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
            pred, meta = translate_constrained(
                self.model,
                self.tokenizer,
                self.trie,
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
            
            return {
                "output": pred,
                "confidence": meta["avg_logprob"],
                "rejected": False
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
