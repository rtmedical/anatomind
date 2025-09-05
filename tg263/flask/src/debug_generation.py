#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug da geração real
"""

import sys
import os
sys.path.append('/workspace/src')

from inference import load_label_set, build_label_trie, build_prefix_allowed_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def debug_generation():
    print("=== DEBUG GERAÇÃO REAL ===")
    
    # Carregar labels
    labels = load_label_set('/workspace/labels/TG263.csv')
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/workspace/model')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Construir trie
    trie = build_label_trie(tokenizer, labels)
    
    # Testar o que realmente acontece no prompt
    text_in = "kidney left"
    prompt = f"Input: {text_in}\nOutput: "
    
    print(f"Prompt: '{prompt}'")
    
    # Tokenização
    enc = tokenizer(prompt, return_tensors="pt")
    prefix_ids = enc["input_ids"][0].tolist()
    
    print(f"Prefix IDs: {prefix_ids}")
    print(f"Último token do prefix: {prefix_ids[-1]} -> '{tokenizer.decode([prefix_ids[-1]])}'")
    
    # Criar função de constrained decoding
    prefix_allowed_tokens_fn = build_prefix_allowed_fn(tokenizer, trie, prefix_ids)
    
    # Testar o que acontece logo após o prompt
    print("\n1. Testando logo após o prompt:")
    input_ids = torch.tensor(prefix_ids)
    allowed = prefix_allowed_tokens_fn(0, input_ids)
    print(f"Tokens permitidos: {len(allowed)}")
    
    # O problema pode ser que está permitindo TODOS os tokens na primeira geração
    # Mas deveria permitir apenas os tokens que estão na raiz do trie
    
    print("\n2. Tokens que deveriam ser permitidos (raiz do trie):")
    root_tokens = list(trie.root.keys())
    print(f"Tokens na raiz: {len(root_tokens)}")
    print(f"Primeiros 10: {root_tokens[:10]}")
    
    # Verificar se 220 (espaço) deveria ser permitido
    print(f"\nToken 220 (espaço) deveria ser permitido? {220 in root_tokens}")
    print(f"Token 32666 (Kid) deveria ser permitido? {32666 in root_tokens}")
    
    # Testar sem o modelo para focar na lógica do trie
    print("\n3. Testando manualmente o que deveria acontecer:")
    
    # Simular adição do primeiro token que está na raiz
    input_ids_with_kid = prefix_ids + [32666]  # " Kid"
    print(f"Após adicionar 32666: {input_ids_with_kid[-5:]}")
    allowed_after_kid = prefix_allowed_tokens_fn(0, torch.tensor(input_ids_with_kid))
    print(f"Tokens permitidos após 'Kid': {len(allowed_after_kid)}")
    
    if len(allowed_after_kid) < 20:
        for token in allowed_after_kid:
            decoded = tokenizer.decode([token])
            print(f"  {token}: '{decoded}'")

if __name__ == "__main__":
    debug_generation()
