# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e

CONFIG_NAME="t5voice_base_libritts_hifitts"
CHECKPOINT_PATH="checkpoints/t5voice_base_libritts_hifitts/checkpoint-pt-250000/model.safetensors"

log() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

usage() {
    echo "Usage: $0 [--config CONFIG_NAME] [--ckpt CHECKPOINT_PATH]"
    echo ""
    echo "Example:"
    echo "  ./export_t5voice.sh \\"
    echo "    --config t5voice_base_libritts_hifitts \\"
    echo "    --ckpt checkpoints/t5voice_base_libritts_hifitts/checkpoint-pt-250000/model.safetensors"
    exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG_NAME="$2"; shift 2;;
    --ckpt) CHECKPOINT_PATH="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "[ERROR] Unknown option: $1"; usage;;
  esac
done

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

log "Exporting T5Voice model"
echo "  Config:     $CONFIG_NAME"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""

python -m t5voice.export.export_t5voice \
    --config-name="$CONFIG_NAME" \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    model.checkpoint_path="$CHECKPOINT_PATH"

log "Export completed successfully"