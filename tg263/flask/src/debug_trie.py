#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de debug para investigar o problema do trie
"""

import sys
import os
sys.path.append('/home/rt/anatomind/tg263/flask/src')

from inference import load_label_set, build_label_trie, build_prefix_allowed_fn
from transformers import AutoTokenizer
import torch

def debug_trie():
    print("=== DEBUG TRIE ===")
    
    # Carregar labels
    print("1. Carregando labels...")
    labels = load_label_set('/workspace/labels/TG263.csv')
    print(f"Total de labels: {len(labels)}")
    
    # Procurar labels relacionadas a kidney
    kidney_labels = [label for label in labels if 'kidney' in label.lower()]
    print(f"\nLabels com 'kidney': {kidney_labels}")
    
    # Carregar tokenizer
    print("\n2. Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('/workspace/model')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Construir trie
    print("\n3. Construindo trie...")
    trie = build_label_trie(tokenizer, labels)
    print("Trie construído!")
    
    # Debug tokenização de algumas labels
    print("\n4. Debug tokenização:")
    test_labels = ['Kidney_L', 'Kidney_R', 'Heart']
    for label in test_labels:
        if label in labels:
            tokens = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
            decoded = tokenizer.decode(tokens)
            print(f"  {label}: {tokens} -> '{decoded}'")
    
    # Testar prefix function
    print("\n5. Testando prefix function...")
    text_in = "kidney left"
    prompt = f"Input: {text_in}\nOutput: "
    
    # Tokenização do prompt
    enc = tokenizer(prompt, return_tensors="pt")
    prefix_ids = enc["input_ids"][0].tolist()
    print(f"Prompt: '{prompt}'")
    print(f"Prefix IDs: {prefix_ids}")
    print(f"Decoded prefix: '{tokenizer.decode(prefix_ids)}'")
    
    # Criar prefix function
    prefix_allowed_tokens_fn = build_prefix_allowed_fn(tokenizer, trie, prefix_ids)
    
    # Testar no início da geração (após o prompt)
    print("\n6. Testando tokens permitidos...")
    
    # Simular IDs após o prompt (apenas o prompt inicial)
    input_ids = torch.tensor(prefix_ids)
    allowed_tokens = prefix_allowed_tokens_fn(0, input_ids)
    print(f"Tokens permitidos após prompt: {len(allowed_tokens)} tokens")
    
    if len(allowed_tokens) == 0:
        print("ERROR: Nenhum token permitido!")
    elif len(allowed_tokens) > 50:
        print(f"Muitos tokens permitidos: {allowed_tokens[:10]}...")
    else:
        print(f"Tokens permitidos: {allowed_tokens}")
    
    # Testar com alguns tokens adicionados
    print("\n7. Testando navegação no trie...")
    
    # Tentar tokenizar o início de "Kidney_L"
    space_token = tokenizer(" ", add_special_tokens=False)["input_ids"]
    print(f"Space token: {space_token}")
    
    k_token = tokenizer("K", add_special_tokens=False)["input_ids"] 
    print(f"'K' token: {k_token}")
    
    kidney_start = tokenizer(" K", add_special_tokens=False)["input_ids"]
    print(f"' K' token: {kidney_start}")
    
    # Simular adição de alguns tokens
    test_sequences = [
        prefix_ids,  # Apenas prompt
        prefix_ids + [29871],  # + espaço (tipico em Llama)
        prefix_ids + [29871, 29968],  # + espaço + K
    ]
    
    for i, seq in enumerate(test_sequences):
        input_ids = torch.tensor(seq)
        allowed = prefix_allowed_tokens_fn(0, input_ids)
        decoded = tokenizer.decode(seq)
        print(f"Seq {i}: {seq[-5:]} -> {len(allowed)} tokens permitidos")
        print(f"  Decoded: '{decoded[-20:]}'")

if __name__ == "__main__":
    debug_trie()
