FROM ghcr.io/huggingface/text-generation-inference:3.0

# Copiar modelo Llama-8B para o container  
COPY model /opt/model/

# Definir diretório de trabalho
WORKDIR /opt

# Configurar cache
ENV HF_HUB_CACHE="/opt/cache"
ENV TRANSFORMERS_CACHE="/opt/cache"

# Configurações específicas para Tesla P40 (Pascal SM 6.1)
ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING="1"

# Desabilitar telemetria e otimizações desnecessárias
ENV TGI_DISABLE_TELEMETRY="1"
ENV TRANSFORMERS_OFFLINE="1"
ENV HF_HUB_OFFLINE="1"
