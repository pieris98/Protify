# Lightweight Docker container for Protify
# Provides a standardized Linux environment for testing and inference with torch.compile support

# 1️⃣  CUDA / cuDNN base with no Python
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 2️⃣  System prerequisites + Python 3.12
# Note: Modal uses Python 3.10, but we use 3.12 for better compatibility
ENV        DEBIAN_FRONTEND=noninteractive \
           PYTHON_VERSION=3.12.7 \
           PATH=/usr/local/bin:$PATH \
           TF_CPP_MIN_LOG_LEVEL=2 \
           TF_ENABLE_ONEDNN_OPTS=0 \
           TOKENIZERS_PARALLELISM=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates wget \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12    /usr/local/bin/pip

# 3️⃣  Location of project code (inside image) – NOT shared with host
WORKDIR /app

# 4️⃣  Copy requirements first for layer caching (matching Modal image order)
COPY requirements.txt .

# Install packages in same order as Modal image:
# 1. Upgrade pip/setuptools
# 2. Install requirements.txt
# 3. Install requirements_modal.txt (if exists)
# 4. Force reinstall torch/torchvision with CUDA 12.8 (last, as in Modal)
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install flash-attn --no-build-isolation && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U

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
