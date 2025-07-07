#!/bin/bash
set -euo pipefail

DIR="$(cd "`dirname "$0"`"/..; pwd)"

# Replace oss ray with offline ve-ray
if [[ "$VE_RAY_OFFLINE_FILE" ]]; then
  echo "Install ve-ray from $VE_RAY_OFFLINE_FILE and overwrite the existing ray"
  pip uninstall -y ray
  pip install "ve-ray[default, data, client]@$VE_RAY_OFFLINE_FILE"
fi
