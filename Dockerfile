FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace
ENV PYTHONPATH="."

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install --system -r pyproject.toml && uv pip install --system runpod>=1.8.1
COPY src/ src/

# Download model while building the image (use CPU for download, CUDA not available during build)
RUN python -c "from src.runpod_handler import download_model; download_model(); print('Model downloaded.')"


CMD ["python", "-u", "src/runpod_handler.py"]