#!/bin/bash
set -euo pipefail

DIR="$(cd "`dirname "$0"`"/..; pwd)"

# Replace oss ray with offline ve-ray
if [[ "$VE_RAY_OFFLINE_FILE" ]]; then
  echo "Install ve-ray from $VE_RAY_OFFLINE_FILE and overwrite the existing ray"
  pip uninstall -y ray
  pip install "ve-ray[default, data, client]@$VE_RAY_OFFLINE_FILE"
fi

# Create and link the cache dir of whisper, the source file might be mounted later
mkdir -p /root/.cache/whisper && \
    ln -s /opt/las/models/iic/speech_whisper-large_lid_multilingual_pytorch/whisper/large-v3.pt /root/.cache/whisper/large-v3.pt
