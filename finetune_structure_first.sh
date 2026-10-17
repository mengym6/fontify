#!/usr/bin/env bash
set -euo pipefail

# runc requires recorded human/delegated lowpass acceptance and user authority
# (inline training_authorization or a bound file). Stage II needs coarse evidence.
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG.json NEW_OUTPUT_DIR [extra structure_first.py options]" >&2
  exit 2
fi
config_path=$1
output_path=$2
shift 2
python_bin=${PYTHON:-python}
process_count=${NPROC_PER_NODE:-1}
if [[ $process_count -gt 1 ]]; then
  "$python_bin" -m torch.distributed.run --standalone \
    --nproc_per_node "$process_count" tools/structure_first.py train \
    --config "$config_path" --output "$output_path" "$@"
else
  "$python_bin" tools/structure_first.py train --config "$config_path" \
    --output "$output_path" "$@"
fi
