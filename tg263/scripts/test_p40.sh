docker run --rm -i --gpus all --entrypoint python \
  -e RUN_TINY=1 \
  ghcr.io/huggingface/text-generation-inference:3.2 <<'PY'
import os, time, torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA não disponível dentro do container"
dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print(f"GPU: {name}  CC: {cc}  Memória: {props.total_memory/1024**3:.1f} GB")

# SDP-Attention: força fallback 'math' (compatível com Pascal)
try:
    from torch.backends.cuda import sdp_kernel
    print("SDPA kernels -> flash:", sdp_kernel.is_flash_available(),
          "mem_efficient:", sdp_kernel.is_mem_efficient_available(),
          "math:", sdp_kernel.is_math_available())
    sdp_kernel.enable_flash(False)
    sdp_kernel.enable_mem_efficient(False)
    sdp_kernel.enable_math(True)
except Exception as e:
    print("SDPA check falhou (ok em Pascal):", e)

# Matmul de fumaça (CUBLAS)
N = 4096
a = torch.randn((N,N), device=dev, dtype=torch.float32)
b = torch.randn((N,N), device=dev, dtype=torch.float32)
torch.cuda.synchronize(); t0 = time.perf_counter()
c = a @ b
torch.cuda.synchronize(); t1 = time.perf_counter()
sec = t1 - t0
gflops = 2*N*N*N / (sec*1e9)
print(f"Matmul {N}x{N}: {sec:.3f}s  ~ {gflops:.1f} GFLOP/s")

# Atenção escalar (fallback math)
from torch.nn.functional import scaled_dot_product_attention as sdpa
q = torch.randn(1, 8, 128, 64, device=dev)
k = torch.randn(1, 8, 128, 64, device=dev)
v = torch.randn(1, 8, 128, 64, device=dev)
out = sdpa(q, k, v, is_causal=False)
print("SDPA (math) OK, saída:", tuple(out.shape))

# Opcional: mini inference com tiny-gpt2 (precisa internet no container)
if os.environ.get("RUN_TINY","0") == "1":
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to(dev)
        ids = tok.encode("Hello from P40:", return_tensors="pt").to(dev)
        gen = model.generate(ids, max_new_tokens=20)
        print("tiny-gpt2 =>", tok.decode(gen[0]))
    except Exception as e:
        print("Transformers test pulado:", e)
PY
