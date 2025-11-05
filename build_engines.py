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
import tensorrt as trt

# -----------------------------
# Helper: Build engine from ONNX
# -----------------------------
def build_engine(onnx_path, engine_path, profiles, precision="fp16", layer_fp32_ops=None):
    """Builds a TensorRT engine from an ONNX model with dynamic shape profiles."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    for profile_shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in profile_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    if layer_fp32_ops:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if any(op in layer.name for op in layer_fp32_ops):
                try:
                    layer.precision = trt.DataType.FLOAT
                except Exception:
                    pass

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build serialized engine from {onnx_path}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Built engine: {engine_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="TensorRT T5Voice Engine Builder")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1)

    parser.add_argument("--enc-min-len", type=int, default=1)
    parser.add_argument("--enc-opt-len", type=int, default=256)
    parser.add_argument("--enc-max-len", type=int, default=512)

    parser.add_argument("--dec-min-past", type=int, default=0)
    parser.add_argument("--dec-opt-past", type=int, default=1024)
    parser.add_argument("--dec-max-past", type=int, default=2048)

    parser.add_argument("--dec-min-len", type=int, default=1)
    parser.add_argument("--dec-opt-len", type=int, default=512)
    parser.add_argument("--dec-max-len", type=int, default=1024)

    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-decoder-layers", type=int, default=12)

    parser.add_argument("--codec-num-codebooks", type=int, default=8)
    parser.add_argument("--codec-min-tokens", type=int, default=16)
    parser.add_argument("--codec-opt-tokens", type=int, default=512)
    parser.add_argument("--codec-max-tokens", type=int, default=4096)

    parser.add_argument("--in-min", type=int, default=1024)
    parser.add_argument("--in-opt", type=int, default=110250)
    parser.add_argument("--in-max", type=int, default=220500)

    args = parser.parse_args()

    print("=========================================")
    print(" Building TensorRT Engines for T5Voice")
    print("-----------------------------------------")
    print(f" Precision        : {args.precision}")
    print(f" Batch size (min/opt/max): {args.min_batch}/{args.opt_batch}/{args.max_batch}")
    print(f" Encoder length   : {args.enc_min_len}/{args.enc_opt_len}/{args.enc_max_len}")
    print(f" Decoder past len : {args.dec_min_past}/{args.dec_opt_past}/{args.dec_max_past}")
    print(f" Codec tokens len : {args.codec_min_tokens}/{args.codec_opt_tokens}/{args.codec_max_tokens}")
    print(f" Codec codebooks  : {args.codec_num_codebooks}")
    print(f" Audio input len  : {args.in_min}/{args.in_opt}/{args.in_max}")
    print("=========================================")

    # -----------------------------
    # 1/4 Encoder
    # -----------------------------
    print("[1/4] Building encoder engine...")
    encoder_profiles = [{
        "input_ids": (
            (args.min_batch, args.enc_min_len),
            (args.opt_batch, args.enc_opt_len),
            (args.max_batch, args.enc_max_len),
        ),
        "attention_mask": (
            (args.min_batch, args.enc_min_len),
            (args.opt_batch, args.enc_opt_len),
            (args.max_batch, args.enc_max_len),
        ),
    }]
    build_engine("t5_encoder.onnx", f"t5_encoder_{args.precision}.engine", encoder_profiles, args.precision, ["ReduceMean", "Pow"])

    # -----------------------------
    # 2/4 Decoder
    # -----------------------------
    print("[2/4] Building decoder engine...")
    decoder_profiles = [{
        "decoder_input_ids": (
            (args.min_batch, args.dec_min_len, 8),
            (args.opt_batch, args.dec_opt_len, 8),
            (args.max_batch, args.dec_max_len, 8),
        ),
        "encoder_hidden_states": (
            (args.min_batch, args.enc_min_len, 768),
            (args.opt_batch, args.enc_opt_len, 768),
            (args.max_batch, args.enc_max_len, 768),
        ),
        "encoder_attention_mask": (
            (args.min_batch, args.enc_min_len),
            (args.opt_batch, args.enc_opt_len),
            (args.max_batch, args.enc_max_len),
        ),
        "decoder_attention_mask": (
            (args.min_batch, args.dec_min_past + args.dec_min_len),
            (args.opt_batch, args.dec_opt_past + args.dec_opt_len),
            (args.max_batch, args.dec_max_past + args.dec_max_len),
        ),
        "past_key_values": (
            (args.min_batch, args.num_decoder_layers, args.num_heads, 2, args.dec_min_past, 64),
            (args.opt_batch, args.num_decoder_layers, args.num_heads, 2, args.dec_opt_past, 64),
            (args.max_batch, args.num_decoder_layers, args.num_heads, 2, args.dec_max_past, 64),
        ),
        "cross_past_key_values": (
            (args.min_batch, args.num_decoder_layers, args.num_heads, 2, args.enc_min_len, 64),
            (args.opt_batch, args.num_decoder_layers, args.num_heads, 2, args.enc_opt_len, 64),
            (args.max_batch, args.num_decoder_layers, args.num_heads, 2, args.enc_max_len, 64),
        ),
    }]
    build_engine("t5_decoder.onnx", f"t5_decoder_{args.precision}.engine", decoder_profiles, args.precision, ["ReduceMean", "Pow"])

    # Always use fp32 for the codec model

    # -----------------------------
    # 3/4 Codec Decoder
    # -----------------------------
    print("[3/4] Building codec decoder engine...")
    codec_decoder_profiles = [{
        "tokens": (
            (args.min_batch, args.codec_num_codebooks, args.codec_min_tokens),
            (args.opt_batch, args.codec_num_codebooks, args.codec_opt_tokens),
            (args.max_batch, args.codec_num_codebooks, args.codec_max_tokens),
        ),
        "tokens_len": (
            (args.min_batch,),
            (args.opt_batch,),
            (args.max_batch,),
        ),
    }]
    build_engine("codec_decode.onnx", f"codec_decode_fp32.engine", codec_decoder_profiles, "fp32")

    # -----------------------------
    # 4/4 Codec Encoder
    # -----------------------------
    print("[4/4] Building codec encoder engine...")
    codec_encoder_profiles = [{
        "audio": (
            (args.min_batch, args.in_min),
            (args.opt_batch, args.in_opt),
            (args.max_batch, args.in_max),
        ),
        "audio_len": (
            (args.min_batch,),
            (args.opt_batch,),
            (args.max_batch,),
        ),
    }]
    build_engine("codec_encode.onnx", f"codec_encode_fp32.engine", codec_encoder_profiles, "fp32")

    print("=========================================")
    print("All TensorRT engines built successfully.")
    print("=========================================")


if __name__ == "__main__":
    main()