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

# LibriTTS Dataset Download and Extract Script
# This script downloads and extracts all LibriTTS dataset splits
# Usage: ./download_libritts.sh [output_directory]

set -e

BASE_URL="https://www.openslr.org/resources/60"

if [ $# -eq 0 ]; then
    OUTPUT_DIR="./libritts"
    echo "No output directory specified. Using default: $OUTPUT_DIR"
else
    OUTPUT_DIR="$1"
    echo "Using specified output directory: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

EXTRACT_DIR="${OUTPUT_DIR}"

SPLITS=(
    "train-clean-100.tar.gz"
    "train-clean-360.tar.gz"
    "train-other-500.tar.gz"
    "dev-clean.tar.gz"
    "dev-other.tar.gz"
    "test-clean.tar.gz"
    "test-other.tar.gz"
)

echo "Starting LibriTTS dataset download..."
echo "Download directory: $OUTPUT_DIR"
echo "Extract directory: $EXTRACT_DIR"
echo "========================================"

for split in "${SPLITS[@]}"; do
    echo ""
    echo "Processing: $split"
    echo "----------------------------------------"
    
    url="${BASE_URL}/${split}"
    output_file="${OUTPUT_DIR}/${split}"
    
    if [ -f "$output_file" ]; then
        echo "File already exists: $output_file"
        echo "Skipping download..."
    else
        echo "Downloading from: $url"
        wget -c "$url" -O "$output_file"
        echo "Download complete: $split"
    fi
    
    echo "Extracting: $split"
    tar -xzf "$output_file" -C "$EXTRACT_DIR"
    echo "Extraction complete: $split"
    
done

echo ""
echo "========================================"
echo "All LibriTTS dataset splits downloaded and extracted successfully!"
echo "Download directory: $OUTPUT_DIR"
echo "Extract directory: $EXTRACT_DIR"
echo ""
echo "Directory structure:"
echo "Archive files (.tar.gz) location: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.tar.gz 2>/dev/null || echo "No archive files found"
echo ""
echo "Extracted dataset location: $EXTRACT_DIR"
tree -L 1 "$EXTRACT_DIR" 2>/dev/null || ls -lh "$EXTRACT_DIR"

echo ""
echo "Total size of extracted data:"
du -sh "$EXTRACT_DIR"
echo ""
echo "Total size including archives:"
du -sh "$OUTPUT_DIR"