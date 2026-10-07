# Chatterbox TTS Server — OpenAI-compatible, zero-shot voice cloning, GPU (CUDA 12.4).
#
# Pinned to a specific upstream commit of Resemble AI's chatterbox (the PyPI package is
# stale and does NOT contain the Multilingual V3 checkpoint — see README). The pinned SHA
# is kept fresh automatically by .github/workflows/sync-upstream.yml.
FROM nvidia/cuda:12.9.2-runtime-ubuntu22.04

# Upstream commit of https://github.com/resemble-ai/chatterbox to build against.
ARG CHATTERBOX_REF=5de7a54aa4

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# PyTorch CUDA 12.4 wheels (matches the torch pin declared by upstream chatterbox).
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Official Chatterbox at the pinned upstream commit (brings its own deps).
RUN pip3 install "git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_REF}"

# Web server layer.
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY server.py /app/server.py

EXPOSE 4123
# CHATTERBOX_T3_MODEL=v3 (default) or v2
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4123"]
