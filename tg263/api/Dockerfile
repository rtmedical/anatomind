# Multi-stage build para API Rust
FROM rust:slim as builder

# Instalar dependências de build
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar workspace
WORKDIR /build

# Copiar manifesto primeiro para cache de dependências
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copiar código fonte
COPY src/ src/
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Instalar runtime dependencies e tini
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -r -s /bin/false anatomind

# Copiar binário
COPY --from=builder /build/target/release/anatomind-api /app/anatomind-api

# Copiar script de healthcheck
COPY --chmod=755 <<'EOF' /app/healthcheck.sh
#!/bin/bash
curl -sf http://localhost:8080/healthz > /dev/null
EOF

# Configurar permissões
RUN chown -R anatomind:anatomind /app
USER anatomind

WORKDIR /app
EXPOSE 8080

# Usar tini como init process
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./anatomind-api"]
