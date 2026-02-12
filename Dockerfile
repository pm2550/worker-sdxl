# base image with cuda 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

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

# Pre-download SDXL weights so the container can run inference offline
RUN python /download_weights.py

CMD ["python", "-u", "/handler.py"]
