FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    media-types \
    libmagic1 \
    ffmpeg \
    ca-certificates \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
ENV PYTHONPATH="."

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install --system -r pyproject.toml && uv pip install --system runpod>=1.7.0
COPY src/ src/

# Default Whisper model to large-v3
ARG WHISPER_MODEL=large-v3
ARG WHISPER_DEVICE=cuda
ARG WHISPER_COMPUTE_TYPE=float16
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV WHISPER_DEVICE=${WHISPER_DEVICE}
ENV WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE}

# Download model while building the image (set LOCAL_FILES_ONLY=false to allow download)
RUN LOCAL_FILES_ONLY=false python -c "from src.runpod_handler import audio2vtt; print('Model downloaded successfully')"

# Set to true for runtime (use cached model only)
ENV LOCAL_FILES_ONLY=true

CMD ["python", "-u", "src/runpod_handler.py"]