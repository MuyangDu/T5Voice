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

# HiFi-TTS Dataset Download and Extract Script
# This script downloads and extracts HiFi-TTS dataset
# Usage: ./download_hifitts.sh [output_directory]

set -e

BASE_URL="https://www.openslr.org/resources/109"

if [ $# -eq 0 ]; then
    OUTPUT_DIR="./hifitts"
    echo "No output directory specified. Using default: $OUTPUT_DIR"
else
    OUTPUT_DIR="$1"
    echo "Using specified output directory: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

EXTRACT_DIR="${OUTPUT_DIR}"

DATASET_FILE="hi_fi_tts_v0.tar.gz"

echo "Starting HiFi-TTS dataset download..."
echo "Download directory: $OUTPUT_DIR"
echo "Extract directory: $EXTRACT_DIR"
echo "========================================"

url="${BASE_URL}/${DATASET_FILE}"
output_file="${OUTPUT_DIR}/${DATASET_FILE}"

echo ""
echo "Processing: $DATASET_FILE"
echo "----------------------------------------"

if [ -f "$output_file" ]; then
    echo "File already exists: $output_file"
    echo "Skipping download..."
else
    echo "Downloading from: $url"
    wget -c "$url" -O "$output_file"
    echo "Download complete: $DATASET_FILE"
fi

echo "Extracting: $DATASET_FILE"
tar -xzf "$output_file" -C "$EXTRACT_DIR"
echo "Extraction complete: $DATASET_FILE"

echo ""
echo "========================================"
echo "HiFi-TTS dataset downloaded and extracted successfully!"
echo "Download directory: $OUTPUT_DIR"
echo "Extract directory: $EXTRACT_DIR"
echo ""
echo "Directory structure:"
echo "Archive file (.tar.gz) location: $OUTPUT_DIR"
ls -lh "$output_file" 2>/dev/null || echo "Archive file not found"
echo ""
echo "Extracted dataset location: $EXTRACT_DIR"
tree -L 2 "$EXTRACT_DIR" 2>/dev/null || ls -lh "$EXTRACT_DIR"

echo ""
echo "Total size of extracted data:"
du -sh "$EXTRACT_DIR"
echo ""
echo "Total size including archive:"
du -sh "$OUTPUT_DIR"