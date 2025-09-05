#!/bin/bash

set -euo pipefail

DIR="$(cd "`dirname "$0"`"/..; pwd)"

PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
UBUNTU_VERSION=${UBUNTU_VERSION:-"24.04"}
CUDA_VERSION=${CUDA_VERSION:-"12.9.1"}
BASE_IMAGE_NAME=${BASE_IMAGE_NAME:-"las-ai-cn-beijing.cr.volces.com/las/las-base"}
BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION:-$(date +%Y%m%d)}
CPU_BASE_IMAGE=${CPU_BASE_IMAGE:-"${BASE_IMAGE_NAME}:py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${BASE_IMAGE_VERSION}"}
GPU_BASE_IMAGE=${GPU_BASE_IMAGE:-"${BASE_IMAGE_NAME}:cu${CUDA_VERSION}-py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${BASE_IMAGE_VERSION}"}

CP_VERSION="cp${PYTHON_VERSION//./}"
FLASH_ATTN_URL=${FLASH_ATTN_URL:-"https://las-ai-cn-beijing.tos-cn-beijing.volces.com/qa/whl/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-${CP_VERSION}-${CP_VERSION}-linux_x86_64.whl"}

if [[ -z "${DAFT_WHEEL_URL+x}" ]]; then
  echo "Daft wheel url is not set, exit"
  exit 1
fi

DAFT_NAME=${DAFT_NAME:-"ve-daft"}
DAFT_VERSION=${DAFT_VERSION:-$(basename "${DAFT_WHEEL_URL}" | awk -F '-' {'print $2'} | sed "s/+/-/")}

VE_RAY_OFFLINE_FILE=${VE_RAY_OFFLINE_FILE:-""}

IMAGE_NAME=${IMAGE_NAME:-"las-ai-cn-beijing.cr.volces.com/las/${DAFT_NAME}"}
CPU_TAG=${CPU_TAG:-"py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${DAFT_VERSION}"}
GPU_TAG=${GPU_TAG:-"cu${CUDA_VERSION}-py${PYTHON_VERSION}-ubuntu${UBUNTU_VERSION}-${DAFT_VERSION}"}


# Build CPU base image
docker build \
  --build-arg BASE_IMAGE="${CPU_BASE_IMAGE}" \
  --build-arg DAFT_WHEEL_URL="${DAFT_WHEEL_URL}" \
  --build-arg DAFT_NAME="${DAFT_NAME}" \
  --build-arg VE_RAY_OFFLINE_FILE="${VE_RAY_OFFLINE_FILE}" \
  --tag "$IMAGE_NAME:$CPU_TAG" \
  --progress=plain \
  --file docker/Dockerfile \
  docker


# Build GPU base image
# Note: only install flash-attention for GPU image
docker build \
  --build-arg BASE_IMAGE="${GPU_BASE_IMAGE}" \
  --build-arg DAFT_WHEEL_URL="${DAFT_WHEEL_URL}" \
  --build-arg DAFT_NAME="${DAFT_NAME}" \
  --build-arg VE_RAY_OFFLINE_FILE="${VE_RAY_OFFLINE_FILE}" \
  --build-arg FLASH_ATTN_URL="${FLASH_ATTN_URL}" \
  --tag "$IMAGE_NAME:$GPU_TAG" \
  --progress=plain \
  --file docker/Dockerfile \
  docker

# push image
docker push "$IMAGE_NAME:$CPU_TAG"
docker push "$IMAGE_NAME:$GPU_TAG"
