# This is iffy, 
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# prevent scripts during boot
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgflags-dev \
    libprotobuf-dev \
    protobuf-compiler \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# link python 
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# install 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

# upgrade pip
RUN python -m pip install --upgrade pip

# prod
FROM base as production

# better layer cache, use req.txt (s)
COPY requirements.txt .
COPY requirements-dev.txt .

# py dependenies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# create code paths
COPY src/ ./src/
COPY configs/ ./configs/
COPY notebooks/ ./notebooks/
COPY README.md .
RUN mkdir -p data/videos data/annotations data/models

ENV PYTHONPATH=/app:$PYTHONPATH
RUN useradd -m -u 1000 basketball && \
    chown -R basketball:basketball /app
USER basketball

# port for Jupiter Notbook
EXPOSE 8888

# Default command
CMD ["python", "-c", "print('container is ready, PS - Docker 9/25')"]

# dev stage
FROM production as development

# go to root to run reqdev
USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    black \
    isort \
    pytest \
    pytest-cov \
    flake8

# bakk to root
USER basketball

# jupiter notbook retart 
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]