#!/usr/bin/env bash
set -euo pipefail
hostname
ls -l /dev/nvidia0 /dev/nvidiactl
${PYTHON_BIN:-/path/to/conda/env/bin/python} -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
