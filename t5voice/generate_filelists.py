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

from tqdm import tqdm
import soundfile as sf
import argparse
import random
import json
import os


def read_file(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines


def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def get_wav_duration(file_path):
    with sf.SoundFile(file_path) as f:
        frames = len(f)
        rate = f.samplerate
        duration = frames / float(rate)
        return duration


def libritts(dataset_dir, codec_dir, train_filelist, test_filelist, dev_filelist):
    codec_dir = os.path.join(dataset_dir, codec_dir)
    train_dirs = ["train-clean-100", "train-clean-360", "train-other-500"]
    test_dirs = ["test-clean", "test-other"]
    dev_dirs = ["dev-clean", "dev-other"]
    train_dirs = [os.path.join(dataset_dir, d) for d in train_dirs]
    test_dirs = [os.path.join(dataset_dir, d) for d in test_dirs]
    dev_dirs = [os.path.join(dataset_dir, d) for d in dev_dirs]

    def find_text_files(directory):
        text_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.normalized.txt'):
                    text_files.append(os.path.join(root, file))
        return text_files
    
    def read_text_file(path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return lines
    
    text_file_paths = []
    for d in train_dirs:
        text_file_paths.extend(find_text_files(d))
    for d in test_dirs:
        text_file_paths.extend(find_text_files(d))
    for d in dev_dirs:
        text_file_paths.extend(find_text_files(d))

    train_output_lines = []
    test_output_lines = []
    dev_output_lines = []

    for text_file_path in tqdm(text_file_paths):
        text = read_file(text_file_path)[0]
        codec_name = os.path.basename(text_file_path).replace(".normalized.txt", ".npy")
        codec_path = os.path.join(codec_dir, codec_name)
        duration_path = codec_path.replace(".npy", ".txt")
        duration = float(read_text_file(duration_path)[0])

        speaker = codec_name.split("_")[0]

        if not os.path.exists(codec_path):
            print(f"{codec_path} not found.")
            continue
            
        json_item = {
            "code": codec_path,
            "text": text,
            "duration": duration,
            "speaker": speaker
        }

        if "test" in text_file_path:
            test_output_lines.append(json.dumps(json_item))
        elif "dev" in text_file_path:
            dev_output_lines.append(json.dumps(json_item))
        else:
            train_output_lines.append(json.dumps(json_item))

    random.shuffle(train_output_lines)
    random.shuffle(test_output_lines)
    random.shuffle(dev_output_lines)
    write_file(train_filelist, train_output_lines)
    write_file(test_filelist, test_output_lines)
    write_file(dev_filelist, dev_output_lines)
    print("Complete!")


def hifitts(dataset_dir, codec_dir, train_filelist, test_filelist, dev_filelist):
    codec_dir = os.path.join(dataset_dir, codec_dir)
    manifests = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    
    def read_text_file(path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return lines
    
    def read_manifest_file(path):
        items = []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line.strip())
            items.append(item)
        return items

    train_output_lines = []
    test_output_lines = []
    dev_output_lines = []

    for manifest in tqdm(manifests):
        manifest_path = os.path.join(dataset_dir, manifest)
        items = read_manifest_file(manifest_path)
        for item in items:
            # print(item)
            text = item["text_normalized"]
            if text[0] == "\"" and text[-1] == "\"":
                text = text[1:-1]
            audio_name = os.path.basename(item["audio_filepath"])
            codec_path = os.path.join(codec_dir, audio_name.replace("flac", "npy"))
            duration_path = codec_path.replace(".npy", ".txt")
            duration = float(read_text_file(duration_path)[0])
            speaker = item["audio_filepath"].split("/")[1].split("_")[0]
            
            if not os.path.exists(codec_path):
                print(f"{codec_path} not found.")
                continue
            
            json_item = {
                "code": codec_path,
                "text": text,
                "duration": duration,
                "speaker": speaker
            }
            
            if "test" in manifest_path:
                test_output_lines.append(json.dumps(json_item))
            elif "dev" in manifest_path:
                dev_output_lines.append(json.dumps(json_item))
            else:
                train_output_lines.append(json.dumps(json_item))

    random.shuffle(train_output_lines)
    random.shuffle(test_output_lines)
    random.shuffle(dev_output_lines)
    write_file(train_filelist, train_output_lines)
    write_file(test_filelist, test_output_lines)
    write_file(dev_filelist, dev_output_lines)
    print("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--codec_dir",
        default="codec",
        type=str,
        required=True
    )
    parser.add_argument(
        "--train_filelist",
        type=str,
        required=True
    )
    parser.add_argument(
        "--test_filelist",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dev_filelist",
        type=str,
        required=False,
        default=""
    )
    
    args = parser.parse_args()
    
    if args.dataset_name == "libritts":
        libritts(args.dataset_dir, args.codec_dir, args.train_filelist, args.test_filelist, args.dev_filelist)
    elif args.dataset_name == "hifitts":
        hifitts(args.dataset_dir, args.codec_dir, args.train_filelist, args.test_filelist, args.dev_filelist)