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

python -m t5voice.inference_tensorrt \
    --config-name=t5voice_base_libritts_hifitts \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    model.checkpoint_path="checkpoints/t5voice_base_libritts_hifitts/checkpoint-pt-250000/model.safetensors" \
    infer.use_logits_processors=true \
    infer.top_k=80 \
    infer.top_p=1.0 \
    infer.temperature=0.85 \
    +reference_audio_path="areference.wav" \
    +reference_audio_text_path="reference.txt" \
    +text_path="text.txt" \
    +output_audio_path="output_t5voice.wav" \
    +max_generation_steps=3000 \
    +streaming=true \
    +chunk_size=50 \
    +overlap_size=2