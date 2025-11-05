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

PYTHON_CMD="python"
CLIENT_SCRIPT="triton_client.py"

SERVER_URL="0.0.0.0:8001"
REF_AUDIO="reference.wav"
REF_TEXT="I’m an assistant here to help with questions, provide information, and support you in various tasks,"
TEXT="I can also offer suggestions, clarify complex topics, and make problem solving easier and more efficient."
OUTPUT="output.wav"

log() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1"
}

usage() {
    echo "Usage: $0 [--url URL] [--ref-audio FILE] [--ref-text TEXT] [--text TEXT] [--output FILE]"
    echo ""
    echo "Example:"
    echo "  ./run_client.sh --url 0.0.0.0:8001 --ref-audio myvoice.wav --ref-text \"Hello there.\" --text \"How are you?\" --output reply.wav"
    exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --url) SERVER_URL="$2"; shift 2;;
    --ref-audio) REF_AUDIO="$2"; shift 2;;
    --ref-text) REF_TEXT="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

if [ ! -f "$CLIENT_SCRIPT" ]; then
    echo "[ERROR] Client script not found: $CLIENT_SCRIPT"
    exit 1
fi

if [ ! -f "$REF_AUDIO" ]; then
    echo "[ERROR] Reference audio file not found: $REF_AUDIO"
    exit 1
fi

log "Running Triton client..."
echo ""
echo "Server URL:     $SERVER_URL"
echo "Reference audio: $REF_AUDIO"
echo "Reference text:  $REF_TEXT"
echo "Input text:      $TEXT"
echo "Output file:     $OUTPUT"
echo ""

$PYTHON_CMD $CLIENT_SCRIPT \
  --url "$SERVER_URL" \
  --ref-audio "$REF_AUDIO" \
  --ref-text "$REF_TEXT" \
  --text "$TEXT" \
  --output "$OUTPUT"

log "Client request completed!"
echo "Generated audio saved to: $OUTPUT"