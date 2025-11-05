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

import json
import numpy as np
import torch
import math
import tensorrt as trt
from tokenize_utils import EnglishIPATokenizer
from vad_utils import trim_silence
import triton_python_backend_utils as pb_utils


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
    


def batched_top_p_filtering(logits: torch.Tensor, top_p: torch.Tensor, min_tokens_to_keep: int = 3) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits."""
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
    """Apply top-k filtering to logits."""
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
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
    
    def infer(self, inputs: dict, output_shapes: dict = None) -> dict:
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            
            if -1 in shape:
                if output_shapes is None or name not in output_shapes:
                    raise RuntimeError(f"Output '{name}' has dynamic shape {shape} but no shape provided")
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


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        parameters = self.model_config.get('parameters', {})
        
        encoder_engine_path = parameters.get('encoder_engine_path', {}).get('string_value', 't5_encoder.engine')
        decoder_engine_path = parameters.get('decoder_engine_path', {}).get('string_value', 't5_decoder.engine')
        codec_encode_engine_path = parameters.get('codec_encode_engine_path', {}).get('string_value', 'codec_encode.engine')
        codec_decode_engine_path = parameters.get('codec_decode_engine_path', {}).get('string_value', 'codec_decode.engine')
        
        language = parameters.get('language', {}).get('string_value', 'en')
        
        self.num_codebooks = int(parameters.get('num_codebooks', {}).get('string_value', '8'))
        self.decoder_vocab_size = int(parameters.get('decoder_vocab_size', {}).get('string_value', '1002'))
        self.num_decoder_special_tokens = int(parameters.get('num_decoder_special_tokens', {}).get('string_value', '2'))
        self.decoder_bos_id = 0
        self.decoder_eos_id = 1
        
        self.chunk_size = int(parameters.get('chunk_size', {}).get('string_value', '50'))
        self.overlap_size = int(parameters.get('overlap_size', {}).get('string_value', '2'))
        self.hop_size = int(parameters.get('hop_size', {}).get('string_value', '256'))
        self.sample_rate = int(parameters.get('sample_rate', {}).get('string_value', '22050'))
        
        self.device = 'cuda'
        
        self.encoder_engine = TensorRTEngine(encoder_engine_path, self.device)
        self.decoder_engine = TensorRTEngine(decoder_engine_path, self.device)
        self.codec_encode_engine = TensorRTEngine(codec_encode_engine_path, self.device)
        self.codec_decode_engine = TensorRTEngine(codec_decode_engine_path, self.device)
        
        if language == "en":
            self.tokenizer = EnglishIPATokenizer()
        else:
            raise ValueError(f"Unsupported language: {language}") 
        
        num_overlap_samples = self.overlap_size * self.hop_size
        self.cross_fade = CrossFade(num_overlap_samples, device=self.device)
    
    def encode_text(self, text: str, ref: bool = False):
        token_ids, _ = self.tokenizer.encode(text, ref=ref)
        return token_ids
    
    def execute(self, requests):
        """
        Execute inference in decoupled mode (streaming).
        """
        responses = []
        
        for request in requests:
            ref_text_tensor = pb_utils.get_input_tensor_by_name(request, "ref_text")
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            ref_audio_tensor = pb_utils.get_input_tensor_by_name(request, "ref_audio")

            ref_text_data = ref_text_tensor.as_numpy()
            ref_text = ref_text_data[0][0].decode('utf-8')

            print(f"Reference Text: {ref_text}")
            
            text_data = text_tensor.as_numpy()
            text = text_data[0][0].decode('utf-8')

            print(f"TTS Text: {text}")
            
            ref_audio = torch.from_numpy(ref_audio_tensor.as_numpy()).to(self.device).float()
            
            temperature = 1.0
            top_k = 50
            top_p = 0.95
            max_length = 3000
            
            temp_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
            if temp_tensor is not None:
                temp_data = temp_tensor.as_numpy()
                temperature = float(temp_data[0][0])
            
            topk_tensor = pb_utils.get_input_tensor_by_name(request, "top_k")
            if topk_tensor is not None:
                topk_data = topk_tensor.as_numpy()
                top_k = int(topk_data[0][0])
            
            topp_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
            if topp_tensor is not None:
                topp_data = topp_tensor.as_numpy()
                top_p = float(topp_data[0][0])
            
            maxlen_tensor = pb_utils.get_input_tensor_by_name(request, "max_length")
            if maxlen_tensor is not None:
                maxlen_data = maxlen_tensor.as_numpy()
                max_length = int(maxlen_data[0][0])
            
            try:
                for audio_chunk in self._inference_stream(
                    ref_audio, ref_text, text,
                    temperature, top_k, top_p, max_length
                ):
                    output_tensor = pb_utils.Tensor("audio_chunk", audio_chunk.cpu().numpy())
                    
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor]
                    )
                    response_sender = request.get_response_sender()
                    response_sender.send(inference_response)
                
                response_sender.send(
                    pb_utils.InferenceResponse(),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                
            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )
                response_sender = request.get_response_sender()
                response_sender.send(
                    error_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        
        return None
    
    def _inference_stream(self, ref_audio, ref_text, text, temperature, top_k, top_p, max_length):
        """Streaming inference implementation."""

        batch_size = 1
        
        ref_input_ids = self.encode_text(ref_text, ref=True)
        input_ids = self.encode_text(text, ref=False)
        
        ref_input_ids = torch.LongTensor(ref_input_ids).to(self.device)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        input_ids[0] = self.tokenizer.tokens.index(" ")
        
        input_ids_with_ref = torch.cat([ref_input_ids[:-1], input_ids]).reshape(batch_size, -1)
        attention_mask = torch.ones_like(input_ids_with_ref)

        ref_audio = trim_silence(ref_audio.cpu().numpy(), self.sample_rate, self.sample_rate, aggressiveness=1)
        ref_audio = torch.tensor(ref_audio, device=self.device)
        
        ref_audio = ref_audio.reshape(batch_size, -1)
        ref_length = torch.tensor([ref_audio.shape[1]], device=self.device)
        
        ref_codec, ref_codec_length = self._codec_encode(ref_audio, ref_length)
        ref_codec = ref_codec[0]
        
        decoder_context_input_ids = ref_codec.new_zeros((ref_codec.shape[0], ref_codec.shape[1] + 1))
        decoder_context_input_ids[:, 1:] = ref_codec + self.num_decoder_special_tokens
        decoder_context_input_ids[:, 0] = self.decoder_bos_id
        decoder_context_input_ids = decoder_context_input_ids.unsqueeze(0).permute(0, 2, 1)
        
        num_overlap_samples = self.overlap_size * self.hop_size
        all_decoder_output_ids = None
        pre_waveform_chunk_tail = torch.zeros((batch_size, num_overlap_samples), device=self.device)
        
        for decoder_output_ids, generated_valid_lengths, is_end in self._generate_stream(
            input_ids_with_ref, attention_mask, decoder_context_input_ids,
            max_length, temperature, top_k, top_p
        ):
            if all_decoder_output_ids is None:
                all_decoder_output_ids = decoder_output_ids
            else:
                decoder_output_ids = decoder_output_ids[:, 1:, :]
                all_decoder_output_ids = torch.cat((all_decoder_output_ids, decoder_output_ids), dim=1)
            
            predicted_codec_tokens = all_decoder_output_ids - self.num_decoder_special_tokens
            predicted_codec_tokens[predicted_codec_tokens < 0] = 0
            predicted_codec_tokens = predicted_codec_tokens.permute(0, 2, 1)
            
            predicted_codec_tokens = predicted_codec_tokens[:, :, -(generated_valid_lengths + self.overlap_size):]
            
            actual_tokens_len = generated_valid_lengths + self.overlap_size
            min_codec_len = self.chunk_size
            
            if actual_tokens_len < min_codec_len:
                padding_len = min_codec_len - actual_tokens_len
                padding = torch.zeros(
                    (batch_size, self.num_codebooks, padding_len),
                    dtype=predicted_codec_tokens.dtype,
                    device=self.device
                )
                predicted_codec_tokens = torch.cat([predicted_codec_tokens, padding], dim=2)
                predicted_codec_tokens_len = torch.tensor([min_codec_len], device=self.device)
            else:
                predicted_codec_tokens_len = torch.tensor([actual_tokens_len], device=self.device)
            
            predicted_waveform = self._codec_decode(predicted_codec_tokens, predicted_codec_tokens_len)
            
            actual_waveform_len = actual_tokens_len * self.hop_size
            predicted_waveform = predicted_waveform[:, :actual_waveform_len]
            
            waveform_chunk_to_respond, pre_waveform_chunk_tail = self.cross_fade(
                predicted_waveform, pre_waveform_chunk_tail
            )
            
            if is_end:
                pre_waveform_chunk_tail *= self.cross_fade.fade_out_coeff
                waveform_chunk_to_respond = torch.cat((waveform_chunk_to_respond, pre_waveform_chunk_tail), dim=1)
            
            yield waveform_chunk_to_respond
    
    def _codec_encode(self, audio, audio_len):
        """Encode audio to codec tokens."""
        batch_size, in_len = audio.shape
        out_len = math.ceil(in_len / self.hop_size)
        
        inputs = {
            'audio': audio.contiguous(),
            'audio_len': audio_len.contiguous()
        }
        
        output_shapes = {
            'tokens': (batch_size, 8, out_len),
            'tokens_len': (batch_size,)
        }
        
        outputs = self.codec_encode_engine.infer(inputs, output_shapes)
        return outputs['tokens'], outputs['tokens_len']
    
    def _codec_decode(self, tokens, tokens_len):
        """Decode codec tokens to audio."""
        batch_size, num_codebooks, seq_len = tokens.shape
        max_samples = seq_len * self.hop_size
        
        inputs = {
            'tokens': tokens.contiguous(),
            'tokens_len': tokens_len.contiguous()
        }
        
        output_shapes = {
            'waveform': (batch_size, max_samples),
            'waveform_lengths': (batch_size,)
        }
        
        outputs = self.codec_decode_engine.infer(inputs, output_shapes)
        return outputs['waveform']
    
    def _generate_stream(self, input_ids, attention_mask, decoder_prompt_input_ids,
                        max_length, temperature, top_k, top_p):
        """Generate tokens in streaming mode."""
        batch_size = input_ids.shape[0]
        
        encoder_outputs = self._encode(input_ids, attention_mask)
        encoder_hidden_states = encoder_outputs['hidden_states']
        cross_past_key_values = encoder_outputs['cross_past_key_values']
        
        decoder_input_ids = decoder_prompt_input_ids
        complete = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        generated_valid_lengths = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        
        past_key_values = torch.zeros(
            (batch_size, 12, 12, 2, 1, 64),
            dtype=torch.float32,
            device=self.device
        )
        
        all_decoder_output_ids = decoder_input_ids.clone()
        chunk_tokens = []

        top_p_tensor = torch.tensor([top_p] * batch_size, device=self.device)
        top_k_tensor = torch.tensor([top_k] * batch_size, device=self.device)
        
        for step in range(max_length):
            current_seq_len = decoder_input_ids.shape[1]
            total_seq_len = past_key_values.shape[4] + current_seq_len
            decoder_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.int64, device=self.device)
            decoder_attention_mask[:, 0] = 0
            
            decoder_outputs = self._decode_step(
                decoder_input_ids, encoder_hidden_states, attention_mask,
                decoder_attention_mask, past_key_values, cross_past_key_values
            )
            
            lm_logits = decoder_outputs['lm_logits']
            present_key_values = decoder_outputs['present_key_values']
            
            logits = lm_logits[:, -1, :, :]
            
            if temperature != 1.0:
                logits = logits / temperature

            if top_p < 1.0:
                logits = batched_top_p_filtering(logits, top_p_tensor, min_tokens_to_keep=3)
            if top_k > 0:
                logits = batched_top_k_filtering(logits, top_k_tensor, min_tokens_to_keep=3)
            
            if generated_valid_lengths[0] < 16:
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
            
            if len(chunk_tokens) == self.chunk_size:
                if complete.all():
                    chunk_tokens = chunk_tokens[:-1]
                    valid_length = len(chunk_tokens)
                    is_end = True
                else:
                    valid_length = self.chunk_size
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
            if valid_length > 0:
                chunk_to_yield = torch.cat(chunk_tokens, dim=1)
                yield chunk_to_yield, valid_length, True
    
    def _encode(self, input_ids, attention_mask):
        """Encode text tokens."""
        inputs = {
            'input_ids': input_ids.contiguous(),
            'attention_mask': attention_mask.contiguous()
        }
        return self.encoder_engine.infer(inputs)
    
    def _decode_step(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask,
                     decoder_attention_mask, past_key_values, cross_past_key_values):
        """Single decoder step."""
        batch_size, cur_seq, num_codebooks = decoder_input_ids.shape
        total_seq = decoder_attention_mask.shape[1]
        
        inputs = {
            'decoder_input_ids': decoder_input_ids.contiguous(),
            'encoder_hidden_states': encoder_hidden_states.contiguous(),
            'encoder_attention_mask': encoder_attention_mask.contiguous(),
            'decoder_attention_mask': decoder_attention_mask.contiguous(),
            'past_key_values': past_key_values.contiguous(),
            'cross_past_key_values': cross_past_key_values.contiguous()
        }
        
        output_shapes = {
            'lm_logits': (batch_size, cur_seq, 8, 1002),
            'present_key_values': (batch_size, 12, 12, 2, total_seq, 64)
        }
        
        return self.decoder_engine.infer(inputs, output_shapes)
    
    def finalize(self):
        """Cleanup resources."""
        pass