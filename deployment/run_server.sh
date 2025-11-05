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
CONTAINER_NAME="t5voice_tritonserver_container"

log() {
    echo -e "\n\033[1;32m[INFO]\033[0m $1"
}

log "Starting Triton Inference Server container..."

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    log "Removing existing container with name $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
fi

docker run -d \
    --gpus all \
    --name $CONTAINER_NAME \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    $IMAGE_NAME:$TAG \
    tritonserver --model-repository=/model_repo

# === Done ===
log "T5Voice Triton Server started successfully."
echo " To view live logs:"
echo "   docker logs -f $CONTAINER_NAME"
echo " To stop and remove the server:"
echo "   docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"