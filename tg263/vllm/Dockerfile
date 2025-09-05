# Dockerfile opcional para vLLM (manter minimal e offline)
FROM rtmedical/anatomind-engine:Llama-8B

# Configurações offline
ENV HF_HUB_OFFLINE=1
ENV VLLM_NO_USAGE_STATS=1

# Manter estrutura original da imagem
# A imagem já deve ter os pesos em /opt/model ou local similar
