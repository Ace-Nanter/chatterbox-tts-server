# OmniVoice TTS Server — OpenAI-compatible, voice cloning + design, GPU (CUDA 12.8).
#
# Pinned to a specific upstream commit of k2-fsa/OmniVoice (upstream publishes no
# usable release tags; master moves fast — see README). The pinned SHA is kept
# fresh automatically by .github/workflows/sync-upstream.yml.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Upstream commit of https://github.com/k2-fsa/OmniVoice to build against.
ARG OMNIVOICE_REF=08be0b4ccbac

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# PyTorch CUDA 12.8 wheels — OmniVoice requires torch>=2.8 (upstream pyproject).
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Official OmniVoice at the pinned upstream commit, with the text-normalization
# extras (WeTextProcessing + num2words): numbers, dates and times ("18 h 30")
# are converted to their spoken form before synthesis.
RUN pip3 install "omnivoice[tn] @ git+https://github.com/k2-fsa/OmniVoice.git@${OMNIVOICE_REF}"

# Web server layer.
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY server.py /app/server.py

EXPOSE 4123
# TTS_DEVICE=cuda (default) or cpu — see README for the full env table.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4123"]
