ARG BASE_IMAGE=las-ai-cn-beijing.cr.volces.com/las/las-base:cu12.9.1-py3.11-ubuntu24.04-20250731
FROM ${BASE_IMAGE}

ARG DAFT_WHEEL_URL
ARG DAFT_NAME=daft
ARG INDEX_URL=https://bytedpypi.byted.org/simple

# pre-download tiktoken for langchain to avoid timeout issue.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache_dir
RUN mkdir -p ${TIKTOKEN_CACHE_DIR}  \
    && wget https://las-ai-cn-beijing.tos-cn-beijing.volces.com/qa/whl/cl100k_base.tiktoken -O ${TIKTOKEN_CACHE_DIR}/9b5ad71b2ce5302211f9c61530b329a4922fc6a4

RUN apt update -y \
    && apt install -y libgl1 libglx-mesa0 iputils-ping telnet imagemagick libreoffice poppler-utils libmagic-dev ffmpeg \
    && apt clean


RUN pip config set global.index-url ${INDEX_URL} \
    && pip install --upgrade pip \
    && pip install "${DAFT_NAME}[las]@${DAFT_WHEEL_URL}"  --resume-retries 3 \
    && pip cache purge

# Used for adding some post logic, e.g. replace ray via ve-ray.
ARG VE_RAY_OFFLINE_FILE
ARG FLASH_ATTN_URL
COPY post_hook.sh /tmp/post_hook.sh
RUN chmod +x /tmp/post_hook.sh && /tmp/post_hook.sh

ENV FORCE_QWENVL_VIDEO_READER=torchvision

WORKDIR /opt/
