FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04  

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    software-properties-common \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 설치
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

# ✅ pip 먼저 설치 (이게 가장 중요)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir sentencepiece

# pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install --no-cache-dir rouge-score sacrebleu
RUN pip install --no-cache-dir sentence-transformers

ENV TOKENIZERS_PARALLELISM=false

# 작업 디렉토리
WORKDIR /app
COPY requirements.txt .

# PyTorch + torchvision 설치
RUN python3 -m pip install --no-cache-dir \
    torch==2.3.1+cu118 \
    torchvision==0.18.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
RUN python3 -m pip install --no-cache-dir -r requirements.txt


