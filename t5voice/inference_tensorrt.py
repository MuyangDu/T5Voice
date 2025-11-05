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

import math
import torch
import numpy as np
import tensorrt as trt
import soundfile as sf
import librosa
from typing import Optional, Tuple
import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm

from t5voice.utils import (
    get_tokenizer,
    get_codec_model,
    update_paths
)
from t5voice.utils.vad_utils import trim_silence

class CrossFade(torch.nn.Module):
    
    def __init__(self, num_overlap_samples, device="cuda"):
        super(CrossFade, self).__init__()
        self.num_overlap_samples = num_overlap_samples
        self.fade_in_coeff, self.fade_out_coeff = self.get_crossfade_coeff()
        self.fade_in_coeff = self.fade_in_coeff.float().to(device)
        self.fade_out_coeff = self.fade_out_coeff.float().to(device)
    
    def get_crossfade_coeff(self):
        fade_len = self.num_overlap_samples
        hann_win = np.hanning(fade_len * 2)
        fade_in = hann_win[:fade_len]
        fade_out = hann_win[fade_len:]
        fade_out_coeff = torch.tensor(fade_out)
        fade_in_coeff = torch.tensor(fade_in)

        return fade_in_coeff, fade_out_coeff
    
    def apply_fade_out(self, waveform):
        waveform_tail = waveform[:, -self.num_overlap_samples:]
        waveform_tail = waveform_tail * self.fade_out_coeff
        waveform = torch.cat((waveform[:, :-self.num_overlap_samples], waveform_tail), dim=1)
        
        return waveform
    
    def apply_fade_in(self, waveform):
        waveform_head = waveform[:, :self.num_overlap_samples]
        waveform_head = waveform_head * self.fade_in_coeff
        waveform = torch.cat((waveform_head, waveform[:, self.num_overlap_samples:]), dim=1)
        
        return waveform
    
    def cross_fade(self, waveform_chunk_to_respond, pre_waveform_chunk_tail):
        waveform_chunk_to_respond = self.apply_fade_in(waveform_chunk_to_respond)
        pre_waveform_chunk_tail = self.apply_fade_out(pre_waveform_chunk_tail)
        waveform_chunk_to_respond_head = waveform_chunk_to_respond[:, :self.num_overlap_samples]
        waveform_chunk_to_respond_head = waveform_chunk_to_respond_head + pre_waveform_chunk_tail[:, -self.num_overlap_samples:]
        waveform_chunk_to_respond = torch.cat((
            waveform_chunk_to_respond_head, waveform_chunk_to_respond[:, self.num_overlap_samples:]), dim=1)
        
        return waveform_chunk_to_respond
    
    def forward(self, waveform_chunk, pre_waveform_chunk_tail):
        cur_waveform_chunk_tail = waveform_chunk[:, -self.num_overlap_samples:]
        waveform_chunk_to_respond = waveform_chunk[:, :-self.num_overlap_samples]
        if waveform_chunk_to_respond.shape[1] < self.num_overlap_samples:
            waveform_chunk_to_respond = pre_waveform_chunk_tail
        else:
            waveform_chunk_to_respond = self.cross_fade(waveform_chunk_to_respond, pre_waveform_chunk_tail)
            
        return waveform_chunk_to_respond, cur_waveform_chunk_tail


def load_audio(path, sample_rate, trim=False):
    if trim:
        y = trim_silence(
            audio=path, 
            target_sample_rate=sample_rate, 
            aggressiveness=1
        )
    else:
        y, sr = librosa.load(path, sr=sample_rate)
    return y


def load_text(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines[0].strip()


def batched_top_p_filtering(logits: torch.Tensor, top_p: torch.Tensor, min_tokens_to_keep: int = 3) -> torch.Tensor:
    batch_size, num_codebooks, vocab_size = logits.shape
    filtered_logits = logits.clone()
    
    for b in range(batch_size):
        for c in range(num_codebooks):
            sorted_logits, sorted_indices = torch.sort(logits[b, c], descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p[b]
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            filtered_logits[b, c, indices_to_remove] = float('-inf')
    
    return filtered_logits


def batched_top_k_filtering(logits: torch.Tensor, top_k: torch.Tensor, min_tokens_to_keep: int = 3) -> torch.Tensor:
    batch_size, num_codebooks, vocab_size = logits.shape
    filtered_logits = logits.clone()
    
    for b in range(batch_size):
        k = max(int(top_k[b].item()), min_tokens_to_keep)
        for c in range(num_codebooks):
            top_k_values, top_k_indices = torch.topk(logits[b, c], k)
            indices_to_remove = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            indices_to_remove[top_k_indices] = False
            filtered_logits[b, c, indices_to_remove] = float('-inf')
    
    return filtered_logits


class TensorRTEngine:
    """Wrapper for TensorRT engine inference."""
    
    _logger = None
    _stream = None
    
    @classmethod
    def get_logger(cls):
        """Get or create shared TensorRT logger."""
        if cls._logger is None:
            cls._logger = trt.Logger(trt.Logger.WARNING)
        return cls._logger
    
    @classmethod
    def get_stream(cls):
        """Get or create shared CUDA stream."""
        if cls._stream is None:
            cls._stream = torch.cuda.Stream()
        return cls._stream
    
    def __init__(self, engine_path: str, device: str = 'cuda'):
        """
        Load TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            device: Device to run on (only 'cuda' supported for TensorRT)
        """
        self.device = device
        self.logger = self.get_logger()
        self.stream = self.get_stream()
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        self.input_names = []
        self.output_names = []
        self.bindings = [None] * self.engine.num_io_tensors
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
    
    def infer(self, inputs: dict, output_shapes: dict = None) -> dict:
        """
        Run inference with TensorRT engine.
        
        Args:
            inputs: Dictionary of input tensors {name: tensor}
            output_shapes: Optional dictionary of output shapes {name: shape}
        
        Returns:
            Dictionary of output tensors {name: tensor}
        """
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            
            if -1 in shape:
                if output_shapes is None or name not in output_shapes:
                    raise RuntimeError(f"Output '{name}' has dynamic shape {shape} but no shape provided in output_shapes")
                shape = output_shapes[name]
            
            dtype = self.engine.get_tensor_dtype(name)
            
            if dtype == trt.DataType.FLOAT:
                torch_dtype = torch.float32
            elif dtype == trt.DataType.HALF:
                torch_dtype = torch.float16
            elif dtype == trt.DataType.INT32:
                torch_dtype = torch.int32
            elif dtype == trt.DataType.INT64:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32
            
            output_tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
            outputs[name] = output_tensor
            self.context.set_tensor_address(name, output_tensor.data_ptr())
        
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        
        return outputs


class T5VoiceTensorRTInference:
    def __init__(
        self, 
        encoder_path: str, 
        decoder_path: str,
        codec_decode_path: Optional[str],
        codec_encode_path: Optional[str],
        codec_model,
        tokenizer,
        num_codebooks: int = 8,
        decoder_vocab_size: int = 1002,
        num_decoder_special_tokens: int = 2,
        device: str = 'cuda'
    ):
        """
        Initialize TensorRT inference for T5Voice model.
        
        Args:
            encoder_path: Path to encoder TensorRT engine
            decoder_path: Path to decoder TensorRT engine
            codec_decode_path: Path to codec decoder TensorRT engine (optional)
            codec_encode_path: Path to codec encoder TensorRT engine (optional)
            codec_model: Audio codec model for encoding/decoding
            tokenizer: Text tokenizer
            num_codebooks: Number of codebooks (default: 8)
            decoder_vocab_size: Decoder vocabulary size (default: 1002)
            num_decoder_special_tokens: Number of special tokens (default: 2)
            device: Device to run on (only 'cuda' supported)
        """
        self.device = device
        assert device == 'cuda', "TensorRT only supports CUDA"
        
        print(f"Loading TensorRT encoder from: {encoder_path}")
        self.encoder_engine = TensorRTEngine(encoder_path, device)
        
        print(f"Loading TensorRT decoder from: {decoder_path}")
        self.decoder_engine = TensorRTEngine(decoder_path, device)
        
        self.use_trt_codec_decode = codec_decode_path is not None
        if self.use_trt_codec_decode:
            print(f"Loading TensorRT codec decoder from: {codec_decode_path}")
            self.codec_decode_engine = TensorRTEngine(codec_decode_path, device)
        
        self.use_trt_codec_encode = codec_encode_path is not None
        if self.use_trt_codec_encode:
            print(f"Loading TensorRT codec encoder from: {codec_encode_path}")
            self.codec_encode_engine = TensorRTEngine(codec_encode_path, device)
        
        self.codec_model = codec_model
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.decoder_vocab_size = decoder_vocab_size
        self.num_decoder_special_tokens = num_decoder_special_tokens
        self.decoder_bos_id = 0
        self.decoder_eos_id = 1
        
        print("\nTensorRT engines loaded successfully!")
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoder inference.
        
        Args:
            input_ids: [batch, seq] - Input token IDs
            attention_mask: [batch, seq] - Attention mask
        
        Returns:
            hidden_states: [batch, seq, 768] - Encoder hidden states
            cross_past_key_values: [batch, 12, 12, 2, seq, 64] - Cross attention KV cache
        """
        inputs = {
            'input_ids': input_ids.contiguous(),
            'attention_mask': attention_mask.contiguous()
        }

        batch_size, seq = input_ids.shape

        output_shapes = {
            'hidden_states': (batch_size, 768),
            'cross_past_key_values': (batch_size, 12, 12, 2, seq, 64)
        }
        
        outputs = self.encoder_engine.infer(inputs, output_shapes)
        
        return outputs['hidden_states'], outputs['cross_past_key_values']
    
    def decode_step(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
        cross_past_key_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run single decoder inference step.
        
        Args:
            decoder_input_ids: [batch, cur_seq, 8] - Decoder input IDs
            encoder_hidden_states: [batch, enc_seq, 768] - Encoder outputs
            encoder_attention_mask: [batch, enc_seq] - Encoder attention mask
            decoder_attention_mask: [batch, total_seq] - Decoder attention mask
            past_key_values: [batch, 12, 12, 2, past_seq, 64] - Self attention KV cache
            cross_past_key_values: [batch, 12, 12, 2, enc_seq, 64] - Cross attention KV cache
        
        Returns:
            lm_logits: [batch, cur_seq, 8, 1002] - Language model logits
            present_key_values: [batch, 12, 12, 2, total_seq, 64] - Updated KV cache
        """
        inputs = {
            'decoder_input_ids': decoder_input_ids.contiguous(),
            'encoder_hidden_states': encoder_hidden_states.contiguous(),
            'encoder_attention_mask': encoder_attention_mask.contiguous(),
            'decoder_attention_mask': decoder_attention_mask.contiguous(),
            'past_key_values': past_key_values.contiguous(),
            'cross_past_key_values': cross_past_key_values.contiguous()
        }

        batch_size, cur_seq, num_codebooks = decoder_input_ids.shape
        total_seq = decoder_attention_mask.shape[1]
        
        output_shapes = {
            'lm_logits': (batch_size, cur_seq, 8, 1002),
            'present_key_values': (batch_size, 12, 12, 2, total_seq, 64)
        }
        
        outputs = self.decoder_engine.infer(inputs, output_shapes)
        
        return outputs['lm_logits'], outputs['present_key_values']
    
    def codec_decode(
        self,
        tokens: torch.Tensor,
        tokens_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run codec decoder inference.
        
        Args:
            tokens: [batch, 8, seq_len] - Codec tokens
            tokens_len: [batch] - Token lengths
        
        Returns:
            waveform: [batch, samples] - Decoded waveform
            waveform_lengths: [batch] - Waveform lengths
        """
        inputs = {
            'tokens': tokens.contiguous(),
            'tokens_len': tokens_len.contiguous()
        }

        batch_size, num_codebooks, seq_len = tokens.shape
        hop_size = 256
        max_samples = seq_len * hop_size
        
        output_shapes = {
            'waveform': (batch_size, max_samples),
            'waveform_lengths': (batch_size,)
        }
        
        outputs = self.codec_decode_engine.infer(inputs, output_shapes)
        
        return outputs['waveform'], outputs['waveform_lengths']
    
    @torch.inference_mode()
    def codec_encode(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run codec encoder inference.
        
        Args:
            audio: [batch, in_len] - Audio waveform
            audio_len: [batch] - Audio lengths
        
        Returns:
            tokens: [batch, 8, out_len] - Codec tokens
            tokens_len: [batch] - Token lengths
        """
        if self.use_trt_codec_encode:
            inputs = {
                'audio': audio.contiguous(),
                'audio_len': audio_len.contiguous()
            }

            batch_size, in_len = audio.shape
            hop_size = 256
            out_len = math.ceil(in_len / hop_size)
            
            output_shapes = {
                'tokens': (batch_size, 8, out_len),
                'tokens_len': (batch_size,)
            }
            
            outputs = self.codec_encode_engine.infer(inputs, output_shapes)
            
            return outputs['tokens'], outputs['tokens_len']
        else:
            tokens, tokens_len = self.codec_model.encode(audio=audio, audio_len=audio_len)
            return tokens, tokens_len
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_prompt_input_ids: Optional[torch.Tensor] = None,
        max_length: int = 1000,
        min_length: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_logits_processors: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio tokens autoregressively.
        
        Args:
            input_ids: [batch, seq] - Input token IDs
            attention_mask: [batch, seq] - Attention mask
            decoder_prompt_input_ids: [batch, prompt_len, num_codebooks] - Optional decoder prompt
            max_length: Maximum generation length
            min_length: Minimum generation length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_logits_processors: Whether to apply top-k/top-p filtering
        
        Returns:
            decoder_output_ids: [batch, seq, num_codebooks] - Generated token IDs
            generated_valid_lengths: [batch, 1] - Valid generation lengths
        """
        batch_size = input_ids.shape[0]
        
        input_ids = input_ids.to(self.device).to(torch.int64)
        attention_mask = attention_mask.to(self.device).to(torch.int64)
        
        if decoder_prompt_input_ids is None:
            decoder_input_ids = torch.zeros(
                (batch_size, 1, self.num_codebooks), 
                dtype=torch.int64, 
                device=self.device
            )
        else:
            decoder_input_ids = decoder_prompt_input_ids.to(self.device).to(torch.int64)
        
        print("Encoding input...")
        encoder_hidden_states, cross_past_key_values = self.encode(input_ids, attention_mask)
        
        complete = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        generated_valid_lengths = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        decoder_output_ids = decoder_input_ids.clone()
        
        past_key_values = torch.zeros(
            (batch_size, 12, 12, 2, 1, 64), 
            dtype=torch.float32, 
            device=self.device
        )
        
        if use_logits_processors:
            top_p_tensor = torch.tensor([top_p] * batch_size, device=self.device)
            top_k_tensor = torch.tensor([top_k] * batch_size, device=self.device)
        
        print(f"Generating (max_length={max_length})...")
        for step in tqdm(range(max_length)):
            current_seq_len = decoder_input_ids.shape[1]
            total_seq_len = past_key_values.shape[4] + current_seq_len
            decoder_attention_mask = torch.ones(
                (batch_size, total_seq_len), 
                dtype=torch.int64, 
                device=self.device
            )
            
            lm_logits, present_key_values = self.decode_step(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                cross_past_key_values=cross_past_key_values
            )
            
            logits = lm_logits[:, -1, :, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if use_logits_processors:
                if top_p < 1.0:
                    logits = batched_top_p_filtering(logits, top_p_tensor, min_tokens_to_keep=3)
                if top_k > 0:
                    logits = batched_top_k_filtering(logits, top_k_tensor, min_tokens_to_keep=3)
            
            if generated_valid_lengths[0] < min_length:
                logits[:, :, self.decoder_eos_id] = float('-inf')
            
            scores = torch.softmax(logits, dim=-1)
            scores = scores.view(-1, self.decoder_vocab_size)
            next_input_ids = torch.multinomial(scores, num_samples=1)
            next_input_ids = next_input_ids.view(-1, 1, self.num_codebooks)
            
            decoder_output_ids = torch.cat([decoder_output_ids, next_input_ids], dim=1)
            decoder_input_ids = next_input_ids
            
            past_key_values = present_key_values
            
            complete = complete | (next_input_ids == self.decoder_eos_id).any(dim=2)
            generated_valid_lengths += (~complete).to(torch.int64)
            
            if complete.all():
                break
        
        decoder_output_ids[:, -1, :] = self.decoder_eos_id
        
        decoder_output_ids = decoder_output_ids[:, 1:, :]
        generated_valid_lengths -= 1
        
        return decoder_output_ids, generated_valid_lengths
    

    @torch.inference_mode()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_prompt_input_ids: Optional[torch.Tensor] = None,
        max_length: int = 1000,
        min_length: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_logits_processors: bool = True,
        chunk_size: int = 50,
        overlap_size: int = 2
    ):
        """
        Generate audio tokens autoregressively with streaming.
        
        Args:
            input_ids: [batch, seq] - Input token IDs
            attention_mask: [batch, seq] - Attention mask
            decoder_prompt_input_ids: [batch, prompt_len, num_codebooks] - Optional decoder prompt
            max_length: Maximum generation length
            min_length: Minimum generation length before allowing EOS
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_logits_processors: Whether to apply top-k/top-p filtering
            chunk_size: Size of each generation chunk
            overlap_size: Number of frames to overlap between chunks
        
        Yields:
            Tuple of (decoder_output_ids, valid_length, is_end)
        """
        batch_size = input_ids.shape[0]
        
        input_ids = input_ids.to(self.device).to(torch.int64)
        attention_mask = attention_mask.to(self.device).to(torch.int64)
        
        if decoder_prompt_input_ids is None:
            decoder_input_ids = torch.zeros(
                (batch_size, 1, self.num_codebooks), 
                dtype=torch.int64, 
                device=self.device
            )
        else:
            decoder_input_ids = decoder_prompt_input_ids.to(self.device).to(torch.int64)
        
        print("Encoding input...")
        encoder_hidden_states, cross_past_key_values = self.encode(input_ids, attention_mask)
        
        complete = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        generated_valid_lengths = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        
        past_key_values = torch.zeros(
            (batch_size, 12, 12, 2, 1, 64), 
            dtype=torch.float32, 
            device=self.device
        )
        
        if use_logits_processors:
            top_p_tensor = torch.tensor([top_p] * batch_size, device=self.device)
            top_k_tensor = torch.tensor([top_k] * batch_size, device=self.device)
        
        all_decoder_output_ids = decoder_input_ids.clone()
        chunk_tokens = []
        
        print(f"Generating with streaming (max_length={max_length}, chunk_size={chunk_size})...")
        for step in tqdm(range(max_length)):
            current_seq_len = decoder_input_ids.shape[1]
            total_seq_len = past_key_values.shape[4] + current_seq_len
            decoder_attention_mask = torch.ones(
                (batch_size, total_seq_len), 
                dtype=torch.int64, 
                device=self.device
            )
            decoder_attention_mask[:, 0] = 0
            
            lm_logits, present_key_values = self.decode_step(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                cross_past_key_values=cross_past_key_values
            )
            
            logits = lm_logits[:, -1, :, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if use_logits_processors:
                if top_p < 1.0:
                    logits = batched_top_p_filtering(logits, top_p_tensor, min_tokens_to_keep=3)
                if top_k > 0:
                    logits = batched_top_k_filtering(logits, top_k_tensor, min_tokens_to_keep=3)
            
            if generated_valid_lengths[0] < min_length:
                logits[:, :, self.decoder_eos_id] = float('-inf')
            
            scores = torch.softmax(logits, dim=-1)
            scores = scores.view(-1, self.decoder_vocab_size)
            next_input_ids = torch.multinomial(scores, num_samples=1)
            next_input_ids = next_input_ids.view(-1, 1, self.num_codebooks)
            
            all_decoder_output_ids = torch.cat([all_decoder_output_ids, next_input_ids], dim=1)
            
            decoder_input_ids = next_input_ids
            
            past_key_values = present_key_values

            complete = complete | (next_input_ids == self.decoder_eos_id).any(dim=2)
            generated_valid_lengths += (~complete).to(torch.int64)
            
            chunk_tokens.append(next_input_ids.clone())
            
            if len(chunk_tokens) == chunk_size:
                if complete.all():
                    chunk_tokens = chunk_tokens[:-1]
                    valid_length = chunk_size - 1
                    is_end = True
                else:
                    valid_length = chunk_size
                    is_end = False
                
                if valid_length > 0:
                    chunk_to_yield = torch.cat(chunk_tokens, dim=1)
                    yield chunk_to_yield, valid_length, is_end
                
                chunk_tokens = []
            
            if complete.all():
                break
        
        if len(chunk_tokens) > 0:
            chunk_tokens = chunk_tokens[:-1]
            valid_length = len(chunk_tokens)
            is_end = True
            
            if valid_length > 0:
                chunk_to_yield = torch.cat(chunk_tokens, dim=1)
                yield chunk_to_yield, valid_length, is_end
    
    @torch.inference_mode()
    def inference(
        self,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_logits_processors: bool = True
    ) -> torch.Tensor:
        """
        T5Voice inference pipeline.
        
        Args:
            ref_audio_path: Path to reference audio
            ref_text: Reference text transcript
            text: Text to synthesize
            max_length: Maximum generation steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            use_logits_processors: Whether to apply sampling filters
        
        Returns:
            predicted_waveform: Generated audio waveform (numpy array)
        """
        ref_input_ids, _ = self.tokenizer.encode(ref_text, ref=True)
        ref_input_ids = torch.LongTensor(ref_input_ids).to(self.device)
        input_ids, _ = self.tokenizer.encode(text)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        batch_size = 1
        
        input_ids[0] = self.tokenizer.tokens.index(" ")
        
        input_ids_with_ref = torch.cat([ref_input_ids[:-1], input_ids]).reshape(batch_size, -1)
        attention_mask = torch.ones_like(input_ids_with_ref)
        
        print(f"Reference Text: {ref_text}")
        print(f"TTS Text: {text}")
        
        ref_audio = load_audio(ref_audio_path, self.codec_model.sample_rate, trim=True)
        ref_audio_tensor = torch.tensor(ref_audio).to(self.device).reshape(batch_size, -1)
        ref_length = torch.tensor(ref_audio_tensor.shape[1]).to(self.device).reshape(batch_size,)
        
        if self.use_trt_codec_encode:
            print("Encoding reference audio with TensorRT codec encoder...")
        else:
            print("Encoding reference audio with PyTorch codec encoder...")
        
        ref_codec, ref_codec_length = self.codec_encode(
            audio=ref_audio_tensor, 
            audio_len=ref_length
        )
        
        ref_codec = ref_codec[0]
        
        decoder_context_input_ids = ref_codec.new_zeros((ref_codec.shape[0], ref_codec.shape[1] + 1))
        decoder_context_input_ids[:, 1:] = ref_codec + self.num_decoder_special_tokens
        decoder_context_input_ids[:, 0] = self.decoder_bos_id
        
        decoder_context_input_ids = decoder_context_input_ids.unsqueeze(0).permute(0, 2, 1)
        
        decoder_output_ids, generated_valid_lengths = self.generate(
            input_ids=input_ids_with_ref,
            attention_mask=attention_mask,
            decoder_prompt_input_ids=decoder_context_input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_logits_processors=use_logits_processors
        )
        
        predicted_codec_tokens = decoder_output_ids - self.num_decoder_special_tokens
        predicted_codec_tokens[predicted_codec_tokens < 0] = 0
        
        predicted_codec_tokens = predicted_codec_tokens.permute(0, 2, 1)
        
        predicted_codec_tokens = predicted_codec_tokens[:, :, ref_codec_length[0]:-1]
        
        if self.use_trt_codec_decode:
            print("Decoding audio with TensorRT codec decoder...")
            predicted_waveform, predicted_waveform_lengths = self.codec_decode(
                tokens=predicted_codec_tokens,
                tokens_len=generated_valid_lengths.squeeze(1)
            )
            actual_len = predicted_waveform_lengths[0].item()
            predicted_waveform = predicted_waveform[0, :actual_len].cpu().numpy()
        else:
            print("Decoding audio with PyTorch codec decoder...")
            predicted_waveform, _ = self.codec_model.decode(
                tokens=predicted_codec_tokens,
                tokens_len=generated_valid_lengths.squeeze(1)
            )
            predicted_waveform = predicted_waveform.detach().squeeze().cpu().numpy()
        
        return predicted_waveform
    

    @torch.inference_mode()
    def inference_stream(
        self,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_logits_processors: bool = True,
        chunk_size: int = 50,
        overlap_size: int = 2
    ) -> torch.Tensor:
        """
        T5Voice inference pipeline with streaming.
        
        Args:
            ref_audio_path: Path to reference audio
            ref_text: Reference text transcript
            text: Text to synthesize
            max_length: Maximum generation steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            use_logits_processors: Whether to apply sampling filters
            chunk_size: Size of each generation chunk
            overlap_size: Number of frames to overlap between chunks
        
        Returns:
            predicted_waveform: Generated audio waveform (numpy array)
        """
        ref_input_ids, _ = self.tokenizer.encode(ref_text, ref=True)
        ref_input_ids = torch.LongTensor(ref_input_ids).to(self.device)
        input_ids, _ = self.tokenizer.encode(text)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        batch_size = 1
        
        input_ids[0] = self.tokenizer.tokens.index(" ")
        
        input_ids_with_ref = torch.cat([ref_input_ids[:-1], input_ids]).reshape(batch_size, -1)
        attention_mask = torch.ones_like(input_ids_with_ref)
        
        print(f"Reference Text: {ref_text}")
        print(f"TTS Text: {text}")
        
        ref_audio = load_audio(ref_audio_path, self.codec_model.sample_rate, trim=True)
        ref_audio_tensor = torch.tensor(ref_audio).to(self.device).reshape(batch_size, -1)
        ref_length = torch.tensor(ref_audio_tensor.shape[1]).to(self.device).reshape(batch_size,)
        
        if self.use_trt_codec_encode:
            print("Encoding reference audio with TensorRT codec encoder...")
        else:
            print("Encoding reference audio with PyTorch codec encoder...")
        
        ref_codec, ref_codec_length = self.codec_encode(
            audio=ref_audio_tensor, 
            audio_len=ref_length
        )
        
        ref_codec = ref_codec[0]
        
        decoder_context_input_ids = ref_codec.new_zeros((ref_codec.shape[0], ref_codec.shape[1] + 1))
        decoder_context_input_ids[:, 1:] = ref_codec + self.num_decoder_special_tokens
        decoder_context_input_ids[:, 0] = self.decoder_bos_id
        
        decoder_context_input_ids = decoder_context_input_ids.unsqueeze(0).permute(0, 2, 1)
        
        hop_size = 256
        num_overlap_frames = overlap_size
        num_overlap_samples = num_overlap_frames * hop_size
        cross_fade = CrossFade(num_overlap_samples, device=self.device)
        
        all_decoder_output_ids = None
        predicted_waveform_chunks = []
        pre_waveform_chunk_tail = torch.zeros((batch_size, num_overlap_samples), device=self.device)
        
        for decoder_output_ids, generated_valid_lengths, is_end in self.generate_stream(
            input_ids=input_ids_with_ref,
            attention_mask=attention_mask,
            decoder_prompt_input_ids=decoder_context_input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_logits_processors=use_logits_processors,
            chunk_size=chunk_size
        ):
            if all_decoder_output_ids is None:
                all_decoder_output_ids = decoder_output_ids
            else:
                decoder_output_ids = decoder_output_ids[:, 1:, :]
                all_decoder_output_ids = torch.cat((all_decoder_output_ids, decoder_output_ids), dim=1)
            
            predicted_codec_tokens = all_decoder_output_ids - self.num_decoder_special_tokens
            predicted_codec_tokens[predicted_codec_tokens < 0] = 0
            predicted_codec_tokens = predicted_codec_tokens.permute(0, 2, 1)
            
            predicted_codec_tokens = predicted_codec_tokens[:, :, -(generated_valid_lengths + num_overlap_frames):]

            actual_tokens_len = generated_valid_lengths + num_overlap_frames
            min_codec_len = chunk_size
            
            if actual_tokens_len < min_codec_len:
                padding_len = min_codec_len - actual_tokens_len
                padding = torch.zeros(
                    (batch_size, self.num_codebooks, padding_len),
                    dtype=predicted_codec_tokens.dtype,
                    device=self.device
                )
                predicted_codec_tokens = torch.cat([predicted_codec_tokens, padding], dim=2)
                predicted_codec_tokens_len = torch.tensor([min_codec_len], device=predicted_codec_tokens.device)
            else:
                predicted_codec_tokens_len = torch.tensor([actual_tokens_len], device=predicted_codec_tokens.device)
            
            
            if self.use_trt_codec_decode:
                predicted_waveform, predicted_waveform_lengths = self.codec_decode(
                    tokens=predicted_codec_tokens,
                    tokens_len=predicted_codec_tokens_len
                )
            else:
                predicted_waveform, predicted_waveform_lengths = self.codec_model.decode(
                    tokens=predicted_codec_tokens,
                    tokens_len=predicted_codec_tokens_len
                )
            
            predicted_waveform = predicted_waveform.detach()

            hop_size = 256
            actual_waveform_len = actual_tokens_len * hop_size
            predicted_waveform = predicted_waveform[:, :actual_waveform_len]
            
            waveform_chunk_to_respond, pre_waveform_chunk_tail = cross_fade(
                predicted_waveform, pre_waveform_chunk_tail
            )
            
            if is_end:
                pre_waveform_chunk_tail *= cross_fade.fade_out_coeff
                waveform_chunk_to_respond = torch.cat((waveform_chunk_to_respond, pre_waveform_chunk_tail), dim=1)
            
            print(f"Chunk Generated, Num Samples: {waveform_chunk_to_respond.shape[1]}")
            
            predicted_waveform_chunks.append(waveform_chunk_to_respond)
        
        predicted_waveform = torch.cat(predicted_waveform_chunks, dim=1).squeeze().cpu().numpy()
        
        return predicted_waveform

@hydra.main(config_path="configs", config_name="t5voice_default", version_base='1.1')
def main(args):
    print(args)
    
    device = "cuda"
    print(f"Using device: {device}")
    
    update_paths(args)
    tokenizer = get_tokenizer(args)
    codec_model = get_codec_model(args)
    codec_model = codec_model.cuda()
    codec_model.eval()
    
    encoder_engine_path = "t5_encoder_fp16.engine"
    decoder_engine_path = "t5_decoder_fp16.engine"
    # Make sure to use fp32 engines for codec encode and decode
    codec_decode_engine_path = "codec_decode_fp32.engine"
    codec_encode_engine_path = "codec_encode_fp32.engine"
    
    trt_inference = T5VoiceTensorRTInference(
        encoder_path=encoder_engine_path,
        decoder_path=decoder_engine_path,
        codec_decode_path=codec_decode_engine_path,
        codec_encode_path=codec_encode_engine_path,
        codec_model=codec_model,
        tokenizer=tokenizer,
        num_codebooks=8,
        device=device
    )
    
    args.reference_audio_path = to_absolute_path(args.reference_audio_path)
    args.reference_audio_text_path = to_absolute_path(args.reference_audio_text_path)
    args.text_path = to_absolute_path(args.text_path)
    args.output_audio_path = to_absolute_path(args.output_audio_path)
    
    ref_text = load_text(args.reference_audio_text_path)
    text = load_text(args.text_path)
    
    streaming = hasattr(args, 'streaming') and args.streaming
    
    if streaming:
        chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else 50
        overlap_size = args.overlap_size if hasattr(args, 'overlap_size') else 2
        
        predicted_waveform = trt_inference.inference_stream(
            ref_audio_path=args.reference_audio_path,
            ref_text=ref_text,
            text=text,
            max_length=args.max_generation_steps,
            temperature=args.infer.temperature,
            top_k=args.infer.top_k,
            top_p=args.infer.top_p,
            use_logits_processors=args.infer.use_logits_processors,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )
    else:
        predicted_waveform = trt_inference.inference(
            ref_audio_path=args.reference_audio_path,
            ref_text=ref_text,
            text=text,
            max_length=args.max_generation_steps,
            temperature=args.infer.temperature,
            top_k=args.infer.top_k,
            top_p=args.infer.top_p,
            use_logits_processors=args.infer.use_logits_processors
        )
    
    sf.write(args.output_audio_path, predicted_waveform, codec_model.sample_rate)
    print(f"Saved to {args.output_audio_path}")

if __name__ == "__main__":
    main()