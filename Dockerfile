# Lightweight Docker container for Protify
# Provides a standardized Linux environment for testing and inference with torch.compile support

# 1️⃣  CUDA / cuDNN base with no Python
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 2️⃣  System prerequisites + Python 3.12
# Note: Modal uses Python 3.10, but we use 3.12 for better compatibility
# Using Ubuntu's pre-built Python 3.12 to avoid memory-intensive compilation
ENV        DEBIAN_FRONTEND=noninteractive \
           PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH \
           CUDA_HOME=/usr/local/cuda \
           TF_CPP_MIN_LOG_LEVEL=2 \
           TF_ENABLE_ONEDNN_OPTS=0 \
           TOKENIZERS_PARALLELISM=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential curl git ca-certificates wget \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        ninja-build procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
# Using PIP_BREAK_SYSTEM_PACKAGES=1 is safe in Docker containers (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -s /usr/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/bin/python3.12 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip setuptools

# 3️⃣  Location of project code (inside image) – NOT shared with host
WORKDIR /app

# 4️⃣  Copy requirements first for layer caching (matching Modal image order)
COPY requirements.txt .

# Install packages in correct order:
# 1. Install requirements.txt first (some packages may install torch, but we'll override it)
# 2. Install torch/torchvision with CUDA 12.8 AFTER requirements (to override any torch version)
# 3. Install flash-attn (slow: compiles CUDA kernels from source)
#    - Automatically detects CPU count and RAM to set optimal MAX_JOBS
#    - ninja-build (installed above) enables parallel compilation, reducing build time from ~2hrs to ~5-10min
RUN pip install -r requirements.txt && \
    pip install ninja && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U && \
    bash -c ' \
      CPU_COUNT=$(nproc) && \
      RAM_GB=$(free -g | awk "/^Mem:/{print \$2}") && \
      RAM_GB=${RAM_GB:-8} && \
      CPU_COUNT=${CPU_COUNT:-2} && \
      echo "Detected: ${CPU_COUNT} CPUs, ${RAM_GB}GB RAM" && \
      MAX_JOBS_BY_CPU=$((CPU_COUNT < 8 ? CPU_COUNT : 8)) && \
      MAX_JOBS_BY_RAM=$((RAM_GB / 8 < 1 ? 1 : RAM_GB / 8)) && \
      MAX_JOBS_BY_RAM=$((MAX_JOBS_BY_RAM > 8 ? 8 : MAX_JOBS_BY_RAM)) && \
      MAX_JOBS=$((MAX_JOBS_BY_CPU < MAX_JOBS_BY_RAM ? MAX_JOBS_BY_CPU : MAX_JOBS_BY_RAM)) && \
      MAX_JOBS=$((MAX_JOBS < 1 ? 1 : MAX_JOBS)) && \
      MAX_JOBS=$((MAX_JOBS > 8 ? 8 : MAX_JOBS)) && \
      echo "Using MAX_JOBS=${MAX_JOBS}, CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} (conservative: ~8GB RAM per job, max 8 jobs)" && \
      MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} pip install flash-attn --no-build-isolation \
    '

# 5️⃣  Copy the rest of the source
COPY . .

# 6️⃣  Change working directory to where the volume will be mounted
WORKDIR /workspace

# 7️⃣  Set working directory to protify source code for running commands
WORKDIR /app/src/protify

# ──────────────────────────────────────────────────────────────────────────────
# 8️⃣  Persistent host volume (/workspace) for caches (HF models, torch cache, etc.)
#     Outputs (logs, results, embeddings, plots, weights) are saved per main.py arguments
#     Bind-mount it when you run the container:  -v ${PWD}:/workspace
# ──────────────────────────────────────────────────────────────────────────────
ENV PROJECT_ROOT=/workspace \
    PYTHONPATH=/app \
    CHAI_DOWNLOADS_DIR=/workspace/models/chai1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    TQDM_CACHE=/workspace/.cache/tqdm

# Only create cache directories in /workspace - outputs (logs, results, embeddings, plots, weights)
# will be created by the code based on arguments passed to main.py
RUN mkdir -p \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/.cache/tqdm \
      /workspace/models/chai1

# Declare the volume so other developers know it's intended to persist
VOLUME ["/workspace"]

# 9️⃣  Default command – override in `docker run … python main.py`
CMD ["bash"]
