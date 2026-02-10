# base image with cuda 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GIN_MODE=release \
    HEALTH_CHECK_INIT_FAST_MODE=true \
    HEALTH_CHECK_INTERVAL_SECONDS=5 \
    SCALING_MIN_QUEUE_TIME_MS=1000 \
    SCALING_THRESHOLD_BUFFER_MS=5000 \
    PIP_NO_CACHE_DIR=1

# keep base image setup simple and stable for RunPod build workers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# install torch separately from the CUDA index for reliable wheel resolution
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.7.0 torchvision==0.22.0

COPY requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt

COPY download_weights.py schemas.py handler.py test_input.json /

# do not pre-download SDXL weights at build time; build-test uses stub mode
CMD python -u /handler.py
