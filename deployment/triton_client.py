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
import numpy as np
import soundfile as sf
import librosa
import queue
import time
from functools import partial
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    """Data structure to collect streaming responses."""
    
    def __init__(self):
        self._chunks = queue.Queue()
        self._chunk_count = 0
        self._complete = False
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._start_request_time = None
        self._error = None

    def mark_start(self):
        self._start_request_time = time.time()
        return self._start_request_time

    def get_num_chunks(self):
        return self._chunk_count

    def get_generation_time(self):
        if self._first_chunk_time and self._last_chunk_time:
            return self._last_chunk_time - self._first_chunk_time
        return 0

    def append(self, chunk):
        self._chunks.put(chunk)
        self._chunk_count += 1

    def mark_first_chunk(self):
        if self._first_chunk_time is None:
            self._first_chunk_time = time.time()

    def mark_last_chunk(self):
        self._complete = True
        self._last_chunk_time = time.time()
    
    def set_error(self, error):
        self._error = error
        self._complete = True
    
    def get_error(self):
        return self._error
    
    def is_complete(self):
        return self._complete
    
    def get_chunks(self):
        """Get all chunks as a list."""
        chunks = []
        while not self._chunks.empty():
            chunks.append(self._chunks.get())
        return chunks


def callback(user_data, result, error):
    """Callback function for handling streaming responses."""
    if error:
        user_data.set_error(error)
        print(f"Error in callback: {error}")
    else:

        if result.get_response().parameters["triton_final_response"].bool_param:
            user_data.mark_last_chunk()
        else:
            user_data.mark_first_chunk()
            audio_chunk = result.as_numpy("audio_chunk")
            user_data.append(audio_chunk)

class TritonT5VoiceClient:
    """Client for T5Voice streaming inference via Triton Inference Server."""
    
    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "t5voice_streaming",
        verbose: bool = False
    ):
        """
        Initialize Triton client.
        
        Args:
            url: Triton server URL (host:port)
            model_name: Name of the model in Triton
            verbose: Enable verbose logging
        """
        self.url = url
        self.model_name = model_name
        self.verbose = verbose
        
        try:
            self.client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose
            )
            
            if not self.client.is_server_live():
                raise Exception("Triton server is not live")
            
            if not self.client.is_server_ready():
                raise Exception("Triton server is not ready")
            
            if not self.client.is_model_ready(model_name):
                raise Exception(f"Model '{model_name}' is not ready")
            
            print(f"Connected to Triton server at {url}")
            print(f"Model '{model_name}' is ready")
            
        except InferenceServerException as e:
            raise Exception(f"Failed to connect to Triton server: {e}")
    
    def load_audio(
        self,
        audio_path: str,
        target_sample_rate: int = 22050
    ) -> np.ndarray:
        """
        Load and resample audio file.
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate
        
        Returns:
            Audio array as float32
        """
        try:
            audio, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
            return audio.astype(np.float32)
        except Exception as e:
            raise Exception(f"Failed to load audio from {audio_path}: {e}")
    
    def inference_stream(
        self,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        output_path: str,
        sample_rate: int = 22050,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 3000
    ):
        """
        Run streaming TTS inference.
        
        Args:
            ref_audio_path: Path to reference audio file
            ref_text: Reference audio transcript
            text: Text to synthesize
            output_path: Path to save output audio
            sample_rate: Audio sample rate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            max_length: Maximum generation steps
        """
        print(f"Loading reference audio from: {ref_audio_path}")
        ref_audio = self.load_audio(ref_audio_path, target_sample_rate=sample_rate)
        print(f"Reference audio shape: {ref_audio.shape}")
        
        print(f"Reference text: {ref_text}")
        print(f"Target text: {text}")
        
        inputs = []
        
        ref_text_data = np.array([[ref_text.encode('utf-8')]], dtype=object)
        ref_text_input = grpcclient.InferInput("ref_text", [1, 1], "BYTES")
        ref_text_input.set_data_from_numpy(ref_text_data)
        inputs.append(ref_text_input)
        
        text_data = np.array([[text.encode('utf-8')]], dtype=object)
        text_input = grpcclient.InferInput("text", [1, 1], "BYTES")
        text_input.set_data_from_numpy(text_data)
        inputs.append(text_input)
        
        ref_audio_input = grpcclient.InferInput("ref_audio", [1, ref_audio.shape[0]], "FP32")
        ref_audio_input.set_data_from_numpy(ref_audio.reshape(1, -1))
        inputs.append(ref_audio_input)
        
        temp_input = grpcclient.InferInput("temperature", [1, 1], "FP32")
        temp_input.set_data_from_numpy(np.array([[temperature]], dtype=np.float32))
        inputs.append(temp_input)
        
        topk_input = grpcclient.InferInput("top_k", [1, 1], "INT32")
        topk_input.set_data_from_numpy(np.array([[top_k]], dtype=np.int32))
        inputs.append(topk_input)
        
        topp_input = grpcclient.InferInput("top_p", [1, 1], "FP32")
        topp_input.set_data_from_numpy(np.array([[top_p]], dtype=np.float32))
        inputs.append(topp_input)
        
        maxlen_input = grpcclient.InferInput("max_length", [1, 1], "INT32")
        maxlen_input.set_data_from_numpy(np.array([[max_length]], dtype=np.int32))
        inputs.append(maxlen_input)
        
        outputs = [grpcclient.InferRequestedOutput("audio_chunk")]
        
        audio_chunks = []
        
        print("\nStarting streaming inference...")
        print("=" * 60)
        
        user_data = UserData()
        start_request_time = user_data.mark_start()
        
        try:
            with grpcclient.InferenceServerClient(url=self.url, verbose=self.verbose) as triton_client:
                triton_client.start_stream(callback=partial(callback, user_data))
                
                triton_client.async_stream_infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs
                )

                last_chunk_count = 0
                while not user_data.is_complete():
                    time.sleep(0.01)
                    
                    current_chunk_count = user_data.get_num_chunks()
                    if current_chunk_count > last_chunk_count:
                        elapsed_time = time.time() - start_request_time
                        print(f"Chunks received: {current_chunk_count} | "
                              f"Time: {elapsed_time:.2f}s", end='\r')
                        last_chunk_count = current_chunk_count
                
                triton_client.stop_stream()
            
            print("\n" + "=" * 60)
            
            if user_data.get_error():
                raise Exception(f"Inference failed: {user_data.get_error()}")
            
            print("Streaming inference completed!")
            
            audio_chunks = user_data.get_chunks()
            
            if len(audio_chunks) == 0:
                raise Exception("No audio chunks received from server")
            
            full_audio = np.concatenate([chunk.squeeze() for chunk in audio_chunks])
            
            print(f"\nSaving audio to: {output_path}")
            sf.write(output_path, full_audio, sample_rate)
            
            total_time = time.time() - start_request_time
            audio_length = full_audio.size / sample_rate
            rtf = total_time / audio_length
            first_chunk_latency = (user_data._first_chunk_time - start_request_time) if user_data._first_chunk_time else 0
            generation_time = user_data.get_generation_time()
            
            print(f"\nStatistics:")
            print(f"  Total chunks: {user_data.get_num_chunks()}")
            print(f"  Total samples: {full_audio.size}")
            print(f"  Audio duration: {audio_length:.2f} seconds")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  First chunk latency: {first_chunk_latency * 1000:.2f} ms")
            print(f"  Generation time: {generation_time:.2f} seconds")
            print(f"  Real-time factor: {rtf:.2f}")
            
        except InferenceServerException as e:
            raise Exception(f"Inference failed: {e}")
        except KeyboardInterrupt:
            print("\n\nInference interrupted by user")
        except Exception as e:
            raise e
    
    def get_server_metadata(self):
        """Get Triton server metadata."""
        try:
            metadata = self.client.get_server_metadata()
            print("\nServer Metadata:")
            print(f"  Name: {metadata.name}")
            print(f"  Version: {metadata.version}")
            print(f"  Extensions: {metadata.extensions}")
        except InferenceServerException as e:
            print(f"Failed to get server metadata: {e}")
    
    def get_model_metadata(self):
        """Get model metadata."""
        try:
            metadata = self.client.get_model_metadata(self.model_name)
            print(f"\nModel Metadata for '{self.model_name}':")
            print(f"  Name: {metadata.name}")
            print(f"  Versions: {metadata.versions}")
            print(f"  Platform: {metadata.platform}")
            
            print("\n  Inputs:")
            for input_meta in metadata.inputs:
                print(f"    - {input_meta.name}: {input_meta.datatype} {input_meta.shape}")
            
            print("\n  Outputs:")
            for output_meta in metadata.outputs:
                print(f"    - {output_meta.name}: {output_meta.datatype} {output_meta.shape}")
                
        except InferenceServerException as e:
            print(f"Failed to get model metadata: {e}")
    
    def get_model_config(self):
        """Get model configuration."""
        try:
            config = self.client.get_model_config(self.model_name)
            print(f"\nModel Config for '{self.model_name}':")
            print(config)
        except InferenceServerException as e:
            print(f"Failed to get model config: {e}")


def main():
    parser = argparse.ArgumentParser(description="T5Voice Streaming TTS Client for Triton")
    
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Triton server URL (default: localhost:8001)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="t5voice",
        help="Model name in Triton (default: t5voice)"
    )
    
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Path to reference audio file"
    )
    
    parser.add_argument(
        "--ref-text",
        type=str,
        required=True,
        help="Reference audio transcript"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=80,
        help="Top-k sampling"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=3000,
        help="Maximum generation steps (default: 3000)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show server and model metadata"
    )
    
    args = parser.parse_args()

    assert args.sample_rate == 22050
    
    try:
        client = TritonT5VoiceClient(
            url=args.url,
            model_name=args.model_name,
            verbose=args.verbose
        )
        
        if args.show_metadata:
            client.get_server_metadata()
            client.get_model_metadata()
            return
        
        client.inference_stream(
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            text=args.text,
            output_path=args.output,
            sample_rate=args.sample_rate,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length
        )
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main() or 0)