#!/bin/bash

set -euo pipefail

DIR="$(cd "`dirname "$0"`"/..; pwd)"

PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
UBUNTU_VERSION=${UBUNTU_VERSION:-"24.04"}
CUDA_VERSION=${CUDA_VERSION:-"12.9.1"}
IMAGE_NAME=${IMAGE_NAME:-"las-ai-qa-cn-beijing.cr.volces.com/las/las-base"}

CUR_DAY=$(date +%Y%m%d)
CPU_TAG=${CPU_TAG:-"py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${CUR_DAY}"}
GPU_TAG=${GPU_TAG:-"cu${CUDA_VERSION}-py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${CUR_DAY}"}

# Build CPU base image
docker build \
    --build-arg BASE_IMAGE="ubuntu:${UBUNTU_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --tag "$IMAGE_NAME:$CPU_TAG" \
    --progress=plain \
    -f docker/Dockerfile-base \
    docker

# Build GPU base image
docker build \
    --build-arg BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --tag "$IMAGE_NAME:$GPU_TAG" \
    --progress=plain \
    -f docker/Dockerfile-base \
    docker

# push image
docker push "$IMAGE_NAME:$CPU_TAG"
docker push "$IMAGE_NAME:$GPU_TAG"
