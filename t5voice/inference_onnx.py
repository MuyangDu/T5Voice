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
import onnxruntime as ort
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


class T5VoiceONNXInference:
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
        Initialize ONNX inference for T5Voice model.
        
        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            codec_model: Audio codec model for encoding/decoding
            tokenizer: Text tokenizer
            num_codebooks: Number of codebooks (default: 8)
            decoder_vocab_size: Decoder vocabulary size (default: 1002)
            num_decoder_special_tokens: Number of special tokens (default: 2)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        self.encoder_session = ort.InferenceSession(
            encoder_path, 
            sess_options=sess_options,
            providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            decoder_path, 
            sess_options=sess_options,
            providers=providers
        )

        self.use_onnx_codec_decode = codec_decode_path is not None
        if self.use_onnx_codec_decode:
            self.codec_decode_session = ort.InferenceSession(
                codec_decode_path,
                sess_options=sess_options,
                providers=providers
            )

        self.use_onnx_codec_encode = codec_encode_path is not None
        if self.use_onnx_codec_encode:
            self.codec_encode_session = ort.InferenceSession(
                codec_encode_path,
                sess_options=sess_options,
                providers=providers
            )
        
        self.codec_model = codec_model
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.decoder_vocab_size = decoder_vocab_size
        self.num_decoder_special_tokens = num_decoder_special_tokens
        self.decoder_bos_id = 0
        self.decoder_eos_id = 1
        
        self._check_execution_providers()
        
        self._get_model_info()
    
    def _check_execution_providers(self):
        """Check which execution providers are being used."""
        encoder_providers = self.encoder_session.get_providers()
        decoder_providers = self.decoder_session.get_providers()
        print(f"Encoder providers: {encoder_providers}")
        print(f"Decoder providers: {decoder_providers}")

        if self.use_onnx_codec_decode:
            codec_decode_providers = self.codec_decode_session.get_providers()
            print(f"Codec Decode providers: {codec_decode_providers}")
        
        if self.use_onnx_codec_encode:
            codec_encode_providers = self.codec_encode_session.get_providers()
            print(f"Codec Encode providers: {codec_encode_providers}")
        
        if self.device == 'cuda' and 'CUDAExecutionProvider' not in encoder_providers:
            print("WARNING: CUDA provider not available for encoder, falling back to CPU")
        if self.device == 'cuda' and 'CUDAExecutionProvider' not in decoder_providers:
            print("WARNING: CUDA provider not available for decoder, falling back to CPU")
        if self.use_onnx_codec_decode and self.device == 'cuda' and 'CUDAExecutionProvider' not in codec_decode_providers:
            print("WARNING: CUDA provider not available for codec decoder, falling back to CPU")
        if self.use_onnx_codec_encode and self.device == 'cuda' and 'CUDAExecutionProvider' not in codec_encode_providers:
            print("WARNING: CUDA provider not available for codec encoder, falling back to CPU")
    
    def _get_model_info(self):
        """Extract model information from ONNX sessions."""
        encoder_inputs = self.encoder_session.get_inputs()
        encoder_outputs = self.encoder_session.get_outputs()
        print("\n=== Encoder Info ===")
        print(f"Inputs: {[(inp.name, inp.shape) for inp in encoder_inputs]}")
        print(f"Outputs: {[(out.name, out.shape) for out in encoder_outputs]}")
        
        decoder_inputs = self.decoder_session.get_inputs()
        decoder_outputs = self.decoder_session.get_outputs()
        print("\n=== Decoder Info ===")
        print(f"Inputs: {[(inp.name, inp.shape) for inp in decoder_inputs]}")
        print(f"Outputs: {[(out.name, out.shape) for out in decoder_outputs]}")

        if self.use_onnx_codec_decode:
            codec_decode_inputs = self.codec_decode_session.get_inputs()
            codec_decode_outputs = self.codec_decode_session.get_outputs()
            print("\n=== Codec Decode Info ===")
            print(f"Inputs: {[(inp.name, inp.shape) for inp in codec_decode_inputs]}")
            print(f"Outputs: {[(out.name, out.shape) for out in codec_decode_outputs]}")
        
        if self.use_onnx_codec_encode:
            codec_encode_inputs = self.codec_encode_session.get_inputs()
            codec_encode_outputs = self.codec_encode_session.get_outputs()
            print("\n=== Codec Encode Info ===")
            print(f"Inputs: {[(inp.name, inp.shape) for inp in codec_encode_inputs]}")
            print(f"Outputs: {[(out.name, out.shape) for out in codec_encode_outputs]}")
    
    def _torch_dtype_to_numpy(self, torch_dtype):
        """Convert torch dtype to numpy dtype."""
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return dtype_map.get(torch_dtype, np.float32)
    
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
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()

        io_binding = self.encoder_session.io_binding()
        
        io_binding.bind_input(
            name='input_ids',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(input_ids.dtype),
            shape=tuple(input_ids.shape),
            buffer_ptr=input_ids.data_ptr()
        )
        
        io_binding.bind_input(
            name='attention_mask',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(attention_mask.dtype),
            shape=tuple(attention_mask.shape),
            buffer_ptr=attention_mask.data_ptr()
        )
        
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.empty(
            (batch_size, seq_len, 768), 
            dtype=torch.float32, 
            device=self.device
        )
        cross_past_key_values = torch.empty(
            (batch_size, 12, 12, 2, seq_len, 64), 
            dtype=torch.float32, 
            device=self.device
        )
        
        io_binding.bind_output(
            name='hidden_states',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(hidden_states.dtype),
            shape=tuple(hidden_states.shape),
            buffer_ptr=hidden_states.data_ptr()
        )
        
        io_binding.bind_output(
            name='cross_past_key_values',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(cross_past_key_values.dtype),
            shape=tuple(cross_past_key_values.shape),
            buffer_ptr=cross_past_key_values.data_ptr()
        )
        
        self.encoder_session.run_with_iobinding(io_binding)
        
        return hidden_states, cross_past_key_values
    
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

        decoder_input_ids = decoder_input_ids.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        encoder_attention_mask = encoder_attention_mask.contiguous()
        decoder_attention_mask = decoder_attention_mask.contiguous()
        past_key_values = past_key_values.contiguous()
        cross_past_key_values = cross_past_key_values.contiguous()

        io_binding = self.decoder_session.io_binding()

        io_binding.bind_input(
            name='decoder_input_ids',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(decoder_input_ids.dtype),
            shape=tuple(decoder_input_ids.shape),
            buffer_ptr=decoder_input_ids.data_ptr()
        )
        
        io_binding.bind_input(
            name='encoder_hidden_states',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(encoder_hidden_states.dtype),
            shape=tuple(encoder_hidden_states.shape),
            buffer_ptr=encoder_hidden_states.data_ptr()
        )
        
        io_binding.bind_input(
            name='encoder_attention_mask',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(encoder_attention_mask.dtype),
            shape=tuple(encoder_attention_mask.shape),
            buffer_ptr=encoder_attention_mask.data_ptr()
        )
        
        io_binding.bind_input(
            name='decoder_attention_mask',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(decoder_attention_mask.dtype),
            shape=tuple(decoder_attention_mask.shape),
            buffer_ptr=decoder_attention_mask.data_ptr()
        )
        
        io_binding.bind_input(
            name='past_key_values',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(past_key_values.dtype),
            shape=tuple(past_key_values.shape),
            buffer_ptr=past_key_values.data_ptr()
        )
        
        io_binding.bind_input(
            name='cross_past_key_values',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(cross_past_key_values.dtype),
            shape=tuple(cross_past_key_values.shape),
            buffer_ptr=cross_past_key_values.data_ptr()
        )
        
        batch_size, cur_seq, _ = decoder_input_ids.shape
        total_seq = past_key_values.shape[4] + cur_seq
        
        lm_logits = torch.empty(
            (batch_size, cur_seq, 8, 1002), 
            dtype=torch.float32, 
            device=self.device
        )
        present_key_values = torch.empty(
            (batch_size, 12, 12, 2, total_seq, 64), 
            dtype=torch.float32, 
            device=self.device
        )
        
        io_binding.bind_output(
            name='lm_logits',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(lm_logits.dtype),
            shape=tuple(lm_logits.shape),
            buffer_ptr=lm_logits.data_ptr()
        )
        
        io_binding.bind_output(
            name='present_key_values',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(present_key_values.dtype),
            shape=tuple(present_key_values.shape),
            buffer_ptr=present_key_values.data_ptr()
        )
        
        self.decoder_session.run_with_iobinding(io_binding)
        
        return lm_logits, present_key_values

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

        tokens = tokens.contiguous()
        tokens_len = tokens_len.contiguous()

        io_binding = self.codec_decode_session.io_binding()
        
        io_binding.bind_input(
            name='tokens',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(tokens.dtype),
            shape=tuple(tokens.shape),
            buffer_ptr=tokens.data_ptr()
        )
        
        io_binding.bind_input(
            name='tokens_len',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(tokens_len.dtype),
            shape=tuple(tokens_len.shape),
            buffer_ptr=tokens_len.data_ptr()
        )
        
        batch_size, num_codebooks, seq_len = tokens.shape
        hop_size = 256
        max_samples = seq_len * hop_size
        
        waveform = torch.empty(
            (batch_size, max_samples), 
            dtype=torch.float32, 
            device=self.device
        )
        waveform_lengths = torch.empty(
            (batch_size,), 
            dtype=torch.int64, 
            device=self.device
        )
        
        io_binding.bind_output(
            name='waveform',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(waveform.dtype),
            shape=tuple(waveform.shape),
            buffer_ptr=waveform.data_ptr()
        )
        
        io_binding.bind_output(
            name='waveform_lengths',
            device_type=self.device,
            device_id=0 if self.device == 'cuda' else 0,
            element_type=self._torch_dtype_to_numpy(waveform_lengths.dtype),
            shape=tuple(waveform_lengths.shape),
            buffer_ptr=waveform_lengths.data_ptr()
        )
        
        self.codec_decode_session.run_with_iobinding(io_binding)
        
        return waveform, waveform_lengths

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
        if self.use_onnx_codec_encode:
            audio = audio.contiguous()
            audio_len = audio_len.contiguous()
            
            io_binding = self.codec_encode_session.io_binding()
            
            io_binding.bind_input(
                name='audio',
                device_type=self.device,
                device_id=0 if self.device == 'cuda' else 0,
                element_type=self._torch_dtype_to_numpy(audio.dtype),
                shape=tuple(audio.shape),
                buffer_ptr=audio.data_ptr()
            )
            
            io_binding.bind_input(
                name='audio_len',
                device_type=self.device,
                device_id=0 if self.device == 'cuda' else 0,
                element_type=self._torch_dtype_to_numpy(audio_len.dtype),
                shape=tuple(audio_len.shape),
                buffer_ptr=audio_len.data_ptr()
            )

            batch_size, in_len = audio.shape
            hop_size = 256
            out_len = math.ceil(in_len / hop_size)
            
            tokens = torch.empty(
                (batch_size, 8, out_len), 
                dtype=torch.int64, 
                device=self.device
            )
            tokens_len = torch.empty(
                (batch_size,), 
                dtype=torch.int64, 
                device=self.device
            )
            
            io_binding.bind_output(
                name='tokens',
                device_type=self.device,
                device_id=0 if self.device == 'cuda' else 0,
                element_type=self._torch_dtype_to_numpy(tokens.dtype),
                shape=tuple(tokens.shape),
                buffer_ptr=tokens.data_ptr()
            )
            
            io_binding.bind_output(
                name='tokens_len',
                device_type=self.device,
                device_id=0 if self.device == 'cuda' else 0,
                element_type=self._torch_dtype_to_numpy(tokens_len.dtype),
                shape=tuple(tokens_len.shape),
                buffer_ptr=tokens_len.data_ptr()
            )
            

            self.codec_encode_session.run_with_iobinding(io_binding)
            
            return tokens, tokens_len
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
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
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
        generated_valid_lengths = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
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
            generated_valid_lengths += (~complete).long()
            
            if complete.all():
                break
        
        decoder_output_ids[:, -1, :] = self.decoder_eos_id
        
        decoder_output_ids = decoder_output_ids[:, 1:, :]
        generated_valid_lengths -= 1
        
        return decoder_output_ids, generated_valid_lengths
    
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

        if self.use_onnx_codec_encode:
            print("Encoding reference audio with ONNX codec encoder...")
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

        if self.use_onnx_codec_decode:
            print("Decoding audio with ONNX codec decoder...")
            predicted_waveform, _ = self.codec_decode(
                tokens=predicted_codec_tokens,
                tokens_len=generated_valid_lengths.squeeze(1)
            )
        else:
            print("Decoding audio with PyTorch codec decoder...")
            predicted_waveform, _ = self.codec_model.decode(
                tokens=predicted_codec_tokens,
                tokens_len=generated_valid_lengths.squeeze(1)
            )
        
        predicted_waveform = predicted_waveform.detach().squeeze().cpu().numpy()
        
        return predicted_waveform


@hydra.main(config_path="configs", config_name="t5voice_default", version_base='1.1')
def main(args):
    print(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    update_paths(args)
    tokenizer = get_tokenizer(args)
    codec_model = get_codec_model(args)
    
    if device == "cuda":
        codec_model = codec_model.cuda()
    codec_model.eval()
    
    encoder_onnx_path = "t5_encoder.onnx"
    decoder_onnx_path = "t5_decoder.onnx"
    codec_decode_onnx_path = "codec_decode.onnx"
    codec_encode_onnx_path = "codec_encode.onnx"
    
    onnx_inference = T5VoiceONNXInference(
        encoder_path=encoder_onnx_path,
        decoder_path=decoder_onnx_path,
        codec_decode_path=codec_decode_onnx_path,
        codec_encode_path=codec_encode_onnx_path,
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
    
    predicted_waveform = onnx_inference.inference(
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