"""
Versão de inferência apenas CPU para isolamento de problemas
"""

import os
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class TG263InferenceCPU:
    def __init__(self, model_path: str, csv_path: str):
        """
        Inicializa o modelo TG-263 apenas em CPU
        """
        self.device = "cpu"  # Força CPU apenas
        logger.info(f"Inicializando TG263InferenceCPU no device: {self.device}")
        
        # Carrega labels
        logger.info(f"Carregando labels TG-263 de: {csv_path}")
        self.labels_df = pd.read_csv(csv_path, sep=';')  # CSV usa ponto e vírgula
        self.tg_labels = self.labels_df['TG263-Primary Name'].tolist()
        logger.info(f"Labels TG-263 carregadas: {len(self.tg_labels)}")
        
        # Carrega modelo e tokenizer
        logger.info(f"Carregando modelo de: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configurações específicas para CPU e estabilidade
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Float32 é mais estável em CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info("Modelo carregado com sucesso!")
        
        # Template de prompt
        self.template = """Given the following anatomical structure description, return the exact TG-263 ID:

Description: {text}
TG-263 ID:"""

    def translate_cpu(self, text: str) -> tuple[str, float]:
        """
        Traduz texto para TG-263 usando apenas CPU com configurações conservadoras
        """
        try:
            # Prepara prompt
            prompt = self.template.format(text=text)
            
            # Tokeniza
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move para CPU (já deveria estar)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Configurações muito conservadoras para geração
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduzido
                    min_new_tokens=1,
                    do_sample=False,    # Determinístico
                    num_beams=1,        # Sem beam search
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False  # Não precisamos de scores
                )
            
            # Decodifica
            generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Limpa output
            output = generated_text.strip()
            if '\n' in output:
                output = output.split('\n')[0].strip()
            
            logger.info(f"Saída bruta do modelo: '{output}'")
            
            # Fuzzy matching para encontrar label mais próximo
            if output and len(output) > 0:
                best_match = process.extractOne(
                    output, 
                    self.tg_labels, 
                    scorer=fuzz.ratio
                )
                
                if best_match and best_match[1] >= 70:  # Score mínimo de 70%
                    confidence = best_match[1] / 100.0
                    logger.info(f"Melhor match: {best_match[0]} (score: {best_match[1]})")
                    return best_match[0], confidence
            
            # Se não encontrou match adequado
            logger.warning(f"Nenhum match adequado encontrado para: '{output}'")
            return "NOT FOUND", -float('inf')
            
        except Exception as e:
            logger.error(f"Erro durante geração CPU: {e}")
            return "NOT FOUND", -float('inf')

    def predict(self, text: str) -> dict:
        """
        Faz predição completa
        """
        logger.info(f"Processando: '{text}'")
        
        output, confidence = self.translate_cpu(text)
        
        # Determina se foi rejeitado
        rejected = confidence < -1.0
        
        result = {
            "output": output,
            "confidence": confidence,
            "rejected": rejected
        }
        
        logger.info(f"Resultado CPU: {result}")
        return result
