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

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra

from t5voice.utils import (
    get_model,
    get_codec_model,
    get_config,
    update_paths
)

class T5EncoderONNXWrapper(nn.Module):
    def __init__(self, t5voice):
        super().__init__()
        self.encoder = t5voice.encoder
        self.decoder = t5voice.decoder
        cfg = t5voice.config
        self.num_decoder_layers = cfg.num_decoder_layers
        self.n_heads = getattr(cfg, "num_heads", None)
        self.key_value_proj_dim = getattr(cfg, "d_kv", None)
        self.head_dim = getattr(cfg, "d_kv", cfg.d_model // self.n_heads)
        

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cross_pkv = []
        for layer in self.decoder.block:
            cross_k = layer.layer[1].EncDecAttention.k(encoder_outputs.hidden_states)
            cross_k = cross_k.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            cross_v = layer.layer[1].EncDecAttention.v(encoder_outputs.hidden_states)
            cross_v = cross_v.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            cross_pkv.append(torch.stack((cross_k, cross_v)))
        
        cross_pkv = torch.stack(cross_pkv)
        cross_pkv = cross_pkv.permute(2, 0, 3, 1, 4, 5)
        cross_pkv = cross_pkv.reshape(batch_size, self.num_decoder_layers, self.n_heads, 2, seq_length, self.head_dim)

        return encoder_outputs.hidden_states, cross_pkv


class T5DecoderONNXWrapper(nn.Module):
    def __init__(self, t5voice):
        super().__init__()
        self.decoder = t5voice.decoder
        self.lm_heads = t5voice.lm_heads
        cfg = t5voice.config
        self.num_layers = cfg.num_decoder_layers
        self.n_heads = getattr(cfg, "num_heads", None)
        self.head_dim = getattr(cfg, "d_kv", cfg.d_model // self.n_heads)

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,       # [B, cur_seq, num_codebooks]
        encoder_hidden_states: torch.FloatTensor,  # [B, enc_seq, d_model]
        encoder_attention_mask: torch.LongTensor,  # [B, enc_seq]
        decoder_attention_mask: torch.LongTensor,  # [B, total_seq]
        past_key_values: torch.FloatTensor,        # [B, L, H, 2, past_seq, D]
        cross_past_key_values: torch.FloatTensor,  # [B, L, H, 2, enc_seq, D]
    ):

        B = decoder_input_ids.size(0)
        L = self.num_layers
        H = self.n_heads
        D = self.head_dim

        past_kv_list = []
        for i in range(L):
            self_k = past_key_values[:, i, :, 0, :, :]
            self_v = past_key_values[:, i, :, 1, :, :]
            cross_k = cross_past_key_values[:, i, :, 0, :, :]
            cross_v = cross_past_key_values[:, i, :, 1, :, :]

            past_kv_list.append((self_k, self_v, cross_k, cross_v))

        past_kv_list = tuple(past_kv_list)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=decoder_attention_mask,
            past_key_values=past_kv_list,
            use_cache=True
        )

        seq_out = decoder_outputs[0]
        present_kv = decoder_outputs.past_key_values

        present_key_values = []
        for (self_k, self_v, cross_k, cross_v) in present_kv:
            present_key_values.append(torch.stack((self_k, self_v)))

        present_key_values = torch.stack(present_key_values)
        present_key_values = present_key_values.permute(2, 0, 3, 1, 4, 5)

        lm_logits = [lm_head(seq_out) for lm_head in self.lm_heads]
        lm_logits = torch.stack(lm_logits)
        lm_logits = lm_logits.permute(1, 2, 0, 3)

        return lm_logits, present_key_values
    

class CodecDecodeONNXWrapper(nn.Module):
    """
    Wraps codec_model.decode(tokens, tokens_len)
    for ONNX export.
    """

    def __init__(self, codec_model):
        super().__init__()
        self.codec_model = codec_model

    def forward(self, tokens: torch.LongTensor, tokens_len: torch.LongTensor):
        """
        tokens: [B, num_codebooks, seq_len]
        tokens_len: [B]
        Returns:
            waveform: [B, samples]
            waveform_lengths: [B]
        """

        waveform, waveform_lengths = self.codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        waveform_lengths = waveform_lengths.long()

        return waveform, waveform_lengths


class CodecEncodeONNXWrapper(nn.Module):

    def __init__(self, codec_model):
        super().__init__()
        self.codec_model = codec_model

    def forward(self, audio: torch.FloatTensor, audio_len: torch.LongTensor):
        """
        Args:
            audio: [B, T]
            audio_len: [B]

        Returns:
            tokens: [B, num_codebooks, seq_len]
            tokens_len: [B]
        """
        tokens, tokens_len = self.codec_model.encode(audio=audio, audio_len=audio_len)
        tokens = tokens.long()
        tokens_len = tokens_len.long()

        return tokens, tokens_len


def export_codec_decoder_onnx(model: nn.Module, out_path: str, device='cuda'):
    """
    Exports the codec decode function to ONNX.
    """
    wrapper = CodecDecodeONNXWrapper(model).to(device).eval()

    batch_size = 1
    num_codebooks = getattr(model, "num_codebooks", 8)
    seq_len = 50

    tokens = torch.randint(
        low=0, high=512,
        size=(batch_size, num_codebooks, seq_len),
        dtype=torch.long, device=device
    )
    tokens_len = torch.tensor([seq_len], dtype=torch.long, device=device)

    torch.onnx.export(
        wrapper,
        (tokens, tokens_len),
        out_path,
        opset_version=17,
        input_names=["tokens", "tokens_len"],
        output_names=["waveform", "waveform_lengths"],
        dynamic_axes={
            "tokens": {0: "batch", 2: "seq_len"},
            "tokens_len": {0: "batch"},
            "waveform": {0: "batch", 1: "audio_len"},
            "waveform_lengths": {0: "batch"}
        },
        do_constant_folding=True,
        verbose=False
    )
    print(f"Saved codec decode ONNX to: {out_path}")


def export_codec_encode_onnx(model: nn.Module, out_path: str, device='cuda'):
    """
    Exports the codec decode function to ONNX.
    """
    wrapper = CodecEncodeONNXWrapper(model).to(device).eval()

    B, T = 1, 2000
    audio = torch.randn((B, T), dtype=torch.float32, device=device)
    audio_len = torch.tensor([T], dtype=torch.long, device=device)

    torch.onnx.export(
        wrapper,
        (audio, audio_len),
        out_path,
        opset_version=17,
        input_names=["audio", "audio_len"],
        output_names=["tokens", "tokens_len"],
        dynamic_axes={
            "audio": {0: "batch", 1: "in_len"},
            "audio_len": {0: "batch"},
            "tokens": {0: "batch", 2: "out_len"},
            "tokens_len": {0: "batch"}
        },
        do_constant_folding=True,
        verbose=False
    )
    print(f"Saved codec decode ONNX to: {out_path}")


def export_t5_encoder_onnx(model: nn.Module, out_path: str, device='cuda'):
    encoder = T5EncoderONNXWrapper(model).to(device).eval()
    B, seq = 1, 100
    input_ids = torch.zeros((B, seq), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, seq), dtype=torch.long, device=device)

    torch.onnx.export(
        encoder,
        (input_ids, attention_mask),
        out_path,
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states", "cross_past_key_values"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "hidden_states": {0: "batch", 1: "seq"},
            "cross_past_key_values": {0: "batch", 4: "seq"}
        },
        verbose=False,
    )
    print(f"Saved encoder ONNX to: {out_path}")

def export_t5_decoder_onnx(model: nn.Module, out_path: str, device='cuda'):
    wrapper = T5DecoderONNXWrapper(model).to(device).eval()
    cfg = model.config
    L = cfg.num_decoder_layers
    n_heads = getattr(cfg, "num_heads", None) or getattr(cfg, "n_heads", None)
    head_dim = getattr(cfg, "d_kv", cfg.d_model // n_heads)

    B, cur_seq, past_seq, enc_seq = 1, 10, 256, 100
    total_seq = cur_seq + past_seq

    decoder_input_ids = torch.zeros((B, cur_seq, cfg.num_codebooks), dtype=torch.long, device=device)
    encoder_hidden_states = torch.randn((B, enc_seq, cfg.d_model), device=device)
    encoder_attention_mask = torch.ones((B, enc_seq), dtype=torch.long, device=device)
    decoder_attention_mask = torch.ones((B, total_seq), dtype=torch.long, device=device)
    decoder_attention_mask[:, 0] = 0
    past_kv = torch.zeros((B, L, n_heads, 2, past_seq, head_dim), dtype=torch.float32, device=device)
    cross_past_kv = torch.zeros((B, L, n_heads, 2, enc_seq, head_dim), dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        (decoder_input_ids, encoder_hidden_states, encoder_attention_mask, decoder_attention_mask, past_kv, cross_past_kv),
        out_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask", "decoder_attention_mask", "past_key_values", "cross_past_key_values"],
        output_names=["lm_logits", "present_key_values"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch", 1: "cur_seq"},
            "encoder_hidden_states": {0: "batch", 1: "enc_seq"},
            "encoder_attention_mask": {0: "batch", 1: "enc_seq"},
            "decoder_attention_mask": {0: "batch", 1: "total_seq"},
            "past_key_values": {0: "batch", 4: "past_seq"},
            "cross_past_key_values": {0: "batch", 4: "enc_seq"},
            "lm_logits": {0: "batch", 1: "cur_seq"},
            "present_key_values": {0: "batch", 4: "total_seq"},
        },
        verbose=False,
    )
    print(f"Saved decoder ONNX to: {out_path}")


def patch_codec_model(codec_model):
    from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer
    def codes(self):
        indices = torch.arange(self.codebook_size)
        indices = rearrange(indices, 'B -> 1 B 1').cuda()
        codes = self.decode(indices=indices, input_len=None)
        codes = codes.squeeze(-1)
        return codes
    FiniteScalarQuantizer.codes = codes

    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    def patched_stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=False,
        )

    def patched_forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        guard = 0 if not self.use_grads else CONSTANT

        real = x[..., 0]
        imag = x[..., 1]
        x = torch.sqrt(real.pow(2) + imag.pow(2) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        if linear_spec:
            return x, seq_len

        x = torch.matmul(self.fb.to(x.dtype), x)
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask

        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)

        return x, seq_len
    
    for name, module in codec_model.named_modules():
        if isinstance(module, FilterbankFeatures):
            module.stft = patched_stft.__get__(module, FilterbankFeatures)
            module.forward = patched_forward.__get__(module, FilterbankFeatures)

    from nemo.collections.tts.models import AudioCodecModel    
    
    def patched_pad_audio(self, audio, audio_len):
        BUFFER_LEN = 1024
        samples_per_frame = self.samples_per_frame

        padded_len = samples_per_frame * torch.ceil(audio_len / samples_per_frame).long()
        max_len = padded_len.max()
        T = audio.shape[1]
        num_padding = max_len - T
        pad_buffer = torch.zeros((audio.size(0), BUFFER_LEN), device=audio.device, dtype=audio.dtype)
        pad_to_take = pad_buffer[:, :num_padding]
        padded_audio = torch.cat([audio, pad_to_take], dim=1)

        return padded_audio, padded_len

    
    AudioCodecModel.pad_audio = patched_pad_audio


@hydra.main(config_path="../configs", config_name="t5voice_default", version_base='1.1')
def main(args):
    print(args)

    device = "cuda"

    update_paths(args)
    config = get_config(args)
    model = get_model(args, config)
    codec_model = get_codec_model(args)

    if device == "cuda":
        model = model.to(device).eval()
        codec_model = codec_model.to(device).eval()

    patch_codec_model(codec_model)

    export_codec_encode_onnx(codec_model, "codec_encode.onnx", device=device)
    export_codec_decoder_onnx(codec_model, "codec_decode.onnx", device=device)
    export_t5_encoder_onnx(model, "t5_encoder.onnx", device=device)
    export_t5_decoder_onnx(model, "t5_decoder.onnx", device=device)

if __name__ == "__main__":
    main()