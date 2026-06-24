# SGLang ROCm Dockerfile for gfx1151 (RDNA 3 / Strix Halo)
# Build:
#   docker build -t sglang-gfx1151 -f docker/gfx1151.Dockerfile .
#
# Run (torch_native backend, recommended):
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#     --group-add=video --ipc=host --network=host \
#     -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
#     sglang-gfx1151 \
#     python3 -m sglang.launch_server \
#       --model Qwen/Qwen3.6-27B \
#       --attention-backend torch_native \
#       --disable-cuda-graph \
#       --trust-remote-code \
#       --host 0.0.0.0 --port 30000
#
# Run (triton backend, no CUDA graphs):
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#     --group-add=video --ipc=host --network=host \
#     -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
#     sglang-gfx1151 \
#     python3 -m sglang.launch_server \
#       --model Qwen/Qwen3.6-27B \
#       --attention-backend triton \
#       --disable-cuda-graph \
#       --trust-remote-code \
#       --host 0.0.0.0 --port 30000

FROM rocm/dev-ubuntu-24.04:7.2.4

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Install system dependencies and create venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    python3-pip \
    python3-dev \
    python3-venv \
    ninja-build \
    cmake \
    pkg-config \
    libopenblas-dev \
    libomp-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /root/.cargo/registry

ENV PATH="/opt/venv/bin:/root/.cargo/bin:$PATH"
ENV CARGO_BUILD_JOBS=4

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch ROCm (matching host: 2.12.1+rocm7.1)
RUN python3 -m pip install --no-cache-dir \
    torch==2.12.1 \
    --index-url https://download.pytorch.org/whl/rocm7.1

# Install Triton (needed by SGLang attention backends)
RUN python3 -m pip install --no-cache-dir triton==3.6.0

# Install flashinfer (Python-only; ROCm uses aiter/triton for GPU kernels)
RUN python3 -m pip install --no-cache-dir flashinfer-python==0.2.3

# Copy local SGLang source (includes feat/gfx1151-rocm-support with torch.zeros fix)
COPY . /opt/sglang

WORKDIR /opt/sglang

# Install SGLang runtime common dependencies
RUN python3 -m pip install --no-cache-dir \
    setuptools_scm \
    IPython \
    aiohttp \
    einops \
    fastapi \
    interegular \
    jinja2 \
    llguidance \
    msgspec \
    numpy \
    orjson \
    outlines==0.1.11 \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client>=0.20.0 \
    psutil \
    py-spy \
    pybase64 \
    pydantic \
    python-multipart \
    pyzmq>=25.1.2 \
    requests \
    scipy \
    sentencepiece \
    setproctitle \
    tiktoken \
    tqdm \
    uvicorn \
    websockets \
    pyyaml

# Install SGLang in editable mode
RUN SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" \
    python3 -m pip install --no-cache-dir -e python

# Re-install ROCm torch (pip install -e python may have pulled CUDA torch)
# and torchvision from ROCm index (CUDA torchvision is incompatible)
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    torch==2.12.1 \
    "torchvision>=0.27.0,<0.28.0" \
    --index-url https://download.pytorch.org/whl/rocm7.1

# Ensure nvidia packages needed by pre-built sgl_kernel persist
RUN python3 -m pip install --no-cache-dir \
    nvidia-cuda-runtime-cu13 nvidia-cuda-nvrtc-cu13 2>/dev/null; exit 0

# Remove kernels package (not needed for inference, conflicts with ROCm)
RUN python3 -m pip uninstall -y kernels kernels-data quack-kernels 2>/dev/null; exit 0

# Clean up pip cache
RUN python3 -m pip cache purge

# Performance environment variables for ROCm
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Default: show help
CMD ["python3", "-m", "sglang.launch_server", "--help"]
