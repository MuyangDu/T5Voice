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

from hydra.utils import to_absolute_path
from functools import wraps
import soundfile as sf
import numpy as np
import hydra
import librosa
import torch
import time

from .utils import (
    get_tokenizer,
    get_model,
    get_codec_model,
    get_config,
    update_paths
)

from .utils.vad_utils import trim_silence


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


@torch.inference_mode()
def t5voice(args, tokenizer, model, codec_model):

    ref_text = load_text(args.reference_audio_text_path)
    text = load_text(args.text_path)

    ref_input_ids, _ = tokenizer.encode(ref_text, ref=True)
    ref_input_ids = torch.LongTensor(ref_input_ids).cuda()
    input_ids, _ = tokenizer.encode(text)
    input_ids = torch.LongTensor(input_ids).cuda()

    batch_size = 1

    input_ids[0] = tokenizer.tokens.index(" ")
    input_ids_with_ref = torch.cat((ref_input_ids[:-1], input_ids)).reshape(batch_size, -1)
    
    print(f"Reference Text: {ref_text}")
    print(f"TTS Text: {text}")

    ref_audio = load_audio(args.reference_audio_path, codec_model.sample_rate, trim=True)
    ref_audio = torch.tensor(ref_audio).cuda().reshape(batch_size, -1)
    ref_length = torch.tensor(ref_audio.shape[1]).cuda().reshape(batch_size,)

    ref_codec, ref_codec_length = codec_model.encode(audio=ref_audio, audio_len=ref_length)
    ref_codec = ref_codec[0]
    decoder_context_input_ids = ref_codec.new_zeros((ref_codec.shape[0], ref_codec.shape[1] + 1))

    num_decoder_special_tokens = 2
    decoder_bos_id = 0

    decoder_context_input_ids[:, 1:] = ref_codec + num_decoder_special_tokens
    decoder_context_input_ids[:, 0] = decoder_bos_id
    # add batch dim
    decoder_context_input_ids = decoder_context_input_ids.unsqueeze(0).permute(0, 2, 1)

    top_p = args.infer.top_p
    top_k = args.infer.top_k
    temperature = args.infer.temperature
    max_generation_steps = args.max_generation_steps
    
    if args.infer.use_logits_processors:
        top_p_tensor = torch.tensor([top_p] * batch_size).cuda()
        top_k_tensor = torch.tensor([top_k] * batch_size).cuda()
        temperature_tensor = torch.tensor([temperature] * batch_size).cuda()
    else:
        top_p_tensor, top_k_tensor, temperature_tensor = None, None, None

    use_cache = args.use_cache
    
    print("Generating...")

    generate_outputs = model.generate(
        input_ids=input_ids_with_ref,
        attention_mask=None,
        decoder_prompt_input_ids=decoder_context_input_ids,
        max_length=max_generation_steps,
        temperature=temperature_tensor,
        top_k=top_k_tensor,
        top_p=top_p_tensor,
        use_cache=use_cache,
        generation_config=model.generation_config,
    )

    decoder_output_ids, generated_valid_lengths, _, _, _, _ = generate_outputs
    predicted_codec_tokens = decoder_output_ids - num_decoder_special_tokens
    predicted_codec_tokens[predicted_codec_tokens < 0] = 0
    total_valid_predicted_lengths = generated_valid_lengths + ref_codec_length - 1
    
    predicted_codec_tokens = predicted_codec_tokens.permute(0, 2, 1)

    predicted_codec_tokens = predicted_codec_tokens[:, :, ref_codec_length: -1]

    predicted_waveform, predicted_waveform_lengths = codec_model.decode(
        tokens=predicted_codec_tokens, 
        tokens_len=generated_valid_lengths.squeeze(1))
    
    predicted_waveform = predicted_waveform.detach().squeeze().cpu().numpy()

    return predicted_waveform


@torch.inference_mode()
def t5voice_stream(args, tokenizer, model, codec_model):
    ref_text = load_text(args.reference_audio_text_path)
    text = load_text(args.text_path)

    ref_input_ids, _ = tokenizer.encode(ref_text, ref=True)
    ref_input_ids = torch.LongTensor(ref_input_ids).cuda()
    input_ids, _ = tokenizer.encode(text)
    input_ids = torch.LongTensor(input_ids).cuda()

    batch_size = 1

    input_ids[0] = tokenizer.tokens.index(" ")
    input_ids_with_ref = torch.cat((ref_input_ids[:-1], input_ids)).reshape(batch_size, -1)
    
    print(f"Reference Audio Text: {ref_text}")
    print(f"TTS Text: {text}")

    ref_audio = load_audio(args.reference_audio_path, codec_model.sample_rate, trim=True)
    ref_audio = torch.tensor(ref_audio).cuda().reshape(batch_size, -1)
    ref_length = torch.tensor(ref_audio.shape[1]).cuda().reshape(batch_size,)

    ref_codec, ref_codec_length = codec_model.encode(audio=ref_audio, audio_len=ref_length)
    ref_codec = ref_codec[0]
    decoder_context_input_ids = ref_codec.new_zeros((ref_codec.shape[0], ref_codec.shape[1] + 1))

    num_decoder_special_tokens = 2
    decoder_bos_id = 0

    decoder_context_input_ids[:, 1:] = ref_codec + num_decoder_special_tokens
    decoder_context_input_ids[:, 0] = decoder_bos_id
    # add batch dim
    decoder_context_input_ids = decoder_context_input_ids.unsqueeze(0).permute(0, 2, 1)

    top_p = args.infer.top_p
    top_k = args.infer.top_k
    temperature = args.infer.temperature
    max_generation_steps = args.max_generation_steps
    
    if args.infer.use_logits_processors:
        top_p_tensor = torch.tensor([top_p] * batch_size).cuda()
        top_k_tensor = torch.tensor([top_k] * batch_size).cuda()
        temperature_tensor = torch.tensor([temperature] * batch_size).cuda()
    else:
        top_p_tensor, top_k_tensor, temperature_tensor = None, None, None

    use_cache = args.use_cache
    
    print("Generating...")

    all_decoder_output_ids = None
    num_overlap_frames = args.overlap_size
    hop_size = 256
    num_overlap_samples = num_overlap_frames * hop_size
    cross_fade = CrossFade(num_overlap_samples)
    total_generated_valid_lengths = 0
    predicted_waveform_chunks = []
    pre_waveform_chunk_tail = torch.zeros((batch_size, num_overlap_samples), device=input_ids.device)

    import pdb

    for decoder_output_ids, generated_valid_lengths, end in model.generate_stream(
        input_ids=input_ids_with_ref,
        attention_mask=None,
        decoder_prompt_input_ids=decoder_context_input_ids,
        max_length=max_generation_steps,
        temperature=temperature_tensor,
        top_k=top_k_tensor,
        top_p=top_p_tensor,
        use_cache=use_cache,
        chunk_size=args.chunk_size,
        generation_config=model.generation_config):

        if all_decoder_output_ids is None:
            all_decoder_output_ids = decoder_output_ids
        else:
            decoder_output_ids = decoder_output_ids[:, 1:, :]
            all_decoder_output_ids = torch.cat((all_decoder_output_ids, decoder_output_ids), dim=1)

        predicted_codec_tokens = all_decoder_output_ids - num_decoder_special_tokens

        total_generated_valid_lengths += generated_valid_lengths
        predicted_codec_tokens[predicted_codec_tokens < 0] = 0
        predicted_codec_tokens = predicted_codec_tokens.permute(0, 2, 1)

        predicted_codec_tokens = predicted_codec_tokens[:, :, -(generated_valid_lengths+num_overlap_frames):]

        if generated_valid_lengths+num_overlap_frames == predicted_codec_tokens.shape[2]:
            predicted_codec_tokens_len = torch.tensor([generated_valid_lengths+num_overlap_frames], device=predicted_codec_tokens.device)
        else:
            predicted_codec_tokens_len = torch.tensor([generated_valid_lengths], device=predicted_codec_tokens.device)

        predicted_waveform, predicted_waveform_lengths = codec_model.decode(
            tokens=predicted_codec_tokens, 
            tokens_len=predicted_codec_tokens_len)

        predicted_waveform = predicted_waveform.detach()

        predicted_waveform_increment = predicted_waveform

        waveform_chunk_to_respond, pre_waveform_chunk_tail = cross_fade(predicted_waveform_increment, pre_waveform_chunk_tail)
        
        if end:
            pre_waveform_chunk_tail *= cross_fade.fade_out_coeff
            waveform_chunk_to_respond = torch.cat((waveform_chunk_to_respond, pre_waveform_chunk_tail), dim=1)
        
        print(f"Chunk Generated, Num Samples: {waveform_chunk_to_respond.shape[1]}")

        predicted_waveform_chunks.append(waveform_chunk_to_respond)
        decoder_context_input_ids = all_decoder_output_ids

    predicted_waveform = torch.cat(predicted_waveform_chunks, dim=1).squeeze().cpu().numpy()

    return predicted_waveform


@hydra.main(config_path="configs", config_name="t5voice_default", version_base='1.1')
def main(args):
    print(args)

    update_paths(args)
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    model = get_model(args, config)
    codec_model = get_codec_model(args)

    model = model.cuda()
    codec_model = codec_model.cuda()
    model.eval()
    codec_model.eval()

    args.reference_audio_path = to_absolute_path(args.reference_audio_path)
    args.reference_audio_text_path = to_absolute_path(args.reference_audio_text_path)
    args.text_path = to_absolute_path(args.text_path)
    args.output_audio_path = to_absolute_path(args.output_audio_path)

    if args.model.klass == "t5voice":
        if args.streaming:
            predicted_waveform = t5voice_stream(args, tokenizer, model, codec_model)
        else:
            predicted_waveform = t5voice(args, tokenizer, model, codec_model)

    sf.write(args.output_audio_path, predicted_waveform, codec_model.sample_rate)

    print(f"Saved to {args.output_audio_path}")


if __name__ == "__main__":
    main()