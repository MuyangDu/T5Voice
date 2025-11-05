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

FROM nvcr.io/nvidia/tritonserver:25.09-py3

WORKDIR /root/build
COPY t5_encoder.onnx t5_decoder.onnx codec_encode.onnx codec_decode.onnx requirements.txt build_engines.py ./

COPY model_repo /model_repo

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /engines && \
    python3 build_engines.py && \
    mv /root/build/*.engine /engines/ || true

ENV ENGINE_DIR=/engines \
    MODEL_REPO=/model_repo

WORKDIR /root
CMD ["bash"]