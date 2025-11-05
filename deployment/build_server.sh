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

IMAGE_NAME="t5voice_tritonserver"
TAG="latest"
DOCKERFILE_PATH="./Dockerfile"

log() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1"
}

log "Starting Docker build for image: ${IMAGE_NAME}:${TAG}"

if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "[ERROR] Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

docker build --network=host -t ${IMAGE_NAME}:${TAG} -f $DOCKERFILE_PATH .

log "Docker image built successfully: ${IMAGE_NAME}:${TAG}"

log "Build completed."