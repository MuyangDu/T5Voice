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

import argparse
from tqdm import tqdm
import tarfile
import librosa
import json
import io
from nemo.collections.tts.models import AudioCodecModel
import torch
import threading
import queue
import time
import numpy as np
import os
from t5voice.utils.vad_utils import trim_silence


def find_wav_files(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_audio(audio_queue, sub_dir_path, sample_rate, pbar):
    try:
        wav_files = find_wav_files(sub_dir_path)
        for wav_file in wav_files:
            y = trim_silence(
                    wav_file, 
                    target_sample_rate=sample_rate, 
                    aggressiveness=3
                )
            audio_queue.put((y, wav_file))
            pbar.update(1)
    except Exception as e:
        print(f"Error: {e}")

def pad_audio_batch(batch_audio, batch_len, batch_max_len):
    padded_audio = torch.zeros((len(batch_audio), batch_max_len), device=batch_audio[0].device)
    for index in range(len(batch_audio)):
        padded_audio[index, :batch_len[index]] = batch_audio[index]
    batch_audio = padded_audio
    batch_len = torch.tensor(batch_len)
    return batch_audio, batch_len

def save_code_batch(encoded_tokens, encoded_len, batch_path, save_base_dir, batch_duration):
    encoded_tokens = encoded_tokens.cpu().numpy()
    encoded_len = encoded_len.cpu().numpy()
    for index in range(len(batch_path)):
        path = batch_path[index]
        audio_name = os.path.basename(path)
        code_name = audio_name[:-3] + "npy"
        duration_name = audio_name[:-3] + "txt"
        code = encoded_tokens[index, :, :encoded_len[index]]
        save_path = os.path.join(save_base_dir, code_name)
        duration_path = os.path.join(save_base_dir, duration_name)
        np.save(save_path, code)
        write_file(duration_path, str(batch_duration[index]))


def generate_code(audio_queue, codec_model, batch_size, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_audio = []
    batch_len = []
    batch_path = []
    batch_duration = []
    batch_max_len = -1

    total = 0

    while True:
        audio, path = audio_queue.get()
        if audio is None and path is None:
            break
        audio_tensor = torch.from_numpy(audio).to(device)
        audio_len = audio_tensor.shape[0]
        batch_audio.append(audio_tensor)
        batch_len.append(audio_len)
        batch_duration.append(audio_len / codec_model.sample_rate)
        batch_path.append(path)
        if audio_len > batch_max_len:
            batch_max_len = audio_len

        if len(batch_audio) == batch_size:
            batch_audio, batch_len = pad_audio_batch(batch_audio, batch_len, batch_max_len)
            batch_len = batch_len.to(device)
            encoded_tokens, encoded_len = codec_model.encode(audio=batch_audio, audio_len=batch_len)
            save_code_batch(encoded_tokens, encoded_len, batch_path, output_dir, batch_duration)
            total += batch_size
            batch_audio = []
            batch_len = []
            batch_path = []
            batch_duration = []
            batch_max_len = -1
        
        audio_queue.task_done()

    if len(batch_audio) > 0:
        batch_audio, batch_len = pad_audio_batch(batch_audio, batch_len, batch_max_len)
        batch_len = batch_len.to(device)
        encoded_tokens, encoded_len = codec_model.encode(audio=batch_audio, audio_len=batch_len)
        save_code_batch(encoded_tokens, encoded_len, batch_path, output_dir, batch_duration)
        total += len(batch_audio)
    print(f"\nComplete! Total processed: {total}")


def wait(q, max_allowed_queue_size=1e3):
    while True:
        if q.qsize() > max_allowed_queue_size:
            time.sleep(1)
        else:
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--codec-model",
        type=str,
        required=True,
        choices=["mel_codec_22khz_medium", "nvidia/low-frame-rate-speech-codec-22khz"],
        default="mel_codec_22khz_medium",
        help="Choose the codec model to use."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        required=True
    )

    args = parser.parse_args()

    batch_size = args.batch_size

    sub_dirs = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "test-clean",
        "test-other",
        "dev-clean",
        "dev-other"
    ]

    print("Counting total wav files...")
    total_wav_files = 0
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(args.dataset_dir, sub_dir)
        if os.path.exists(sub_dir_path):
            wav_files = find_wav_files(sub_dir_path)
            total_wav_files += len(wav_files)
    print(f"Found {total_wav_files} wav files in total")

    codec_model = AudioCodecModel.from_pretrained(args.codec_model)
    codec_model.freeze()

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    audio_queue = queue.Queue()
    
    pbar = tqdm(total=total_wav_files, desc="Processing audio files", unit="file")
    
    generate_code_thread = threading.Thread(target=generate_code, args=(audio_queue, codec_model, batch_size, output_dir))
    generate_code_thread.start()
    
    load_audio_threads = []
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(args.dataset_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            continue
        load_audio_thread = threading.Thread(target=load_audio, args=(audio_queue, sub_dir_path, codec_model.sample_rate, pbar))
        load_audio_threads.append(load_audio_thread)
        load_audio_thread.start()

        if len(load_audio_threads) >= 1:
            for thread in load_audio_threads:
                thread.join()
            load_audio_threads = []
    
    for thread in load_audio_threads:
        thread.join()

    audio_queue.put((None, None))

    generate_code_thread.join()
    
    pbar.close()