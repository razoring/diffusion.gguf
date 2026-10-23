import os
import sys
import struct
import numpy as np
import subprocess

sys.path.insert(0, os.path.abspath("third_party/llama.cpp/gguf-py"))
import gguf

CONVERT_BIN = os.path.abspath("build/bin/Release/diffusion-convert.exe")
CLI_BIN = os.path.abspath("build/bin/Release/diffusion-cli.exe")
MODEL_SRC = "models/stories15M-q4_0.gguf"

TENSOR_MAP_REVERSE = {
    "token_embd.weight": "tok_embeddings.weight",
    "output_norm.weight": "norm.weight",
    "output.weight": "output.weight",
}

def map_to_ggjt_name(name):
    if name in TENSOR_MAP_REVERSE:
        return TENSOR_MAP_REVERSE[name]
    if name.startswith("blk."):
        parts = name.split(".")
        layer_num = parts[1]
        suffix = ".".join(parts[2:])
        name_map = {
            "attn_q.weight": "attention.wq.weight",
            "attn_k.weight": "attention.wk.weight",
            "attn_v.weight": "attention.wv.weight",
            "attn_output.weight": "attention.wo.weight",
            "attn_norm.weight": "attention_norm.weight",
            "ffn_gate.weight": "feed_forward.w1.weight",
            "ffn_down.weight": "feed_forward.w2.weight",
            "ffn_up.weight": "feed_forward.w3.weight",
            "ffn_norm.weight": "ffn_norm.weight",
        }
        if suffix in name_map:
            return f"layers.{layer_num}.{name_map[suffix]}"
    return name

def make_ggjt(source_gguf_path, out_ggjt_path, version=3):
    reader = gguf.GGUFReader(source_gguf_path)

    tf = reader.get_field("tokenizer.ggml.tokens")
    sf = reader.get_field("tokenizer.ggml.scores")
    tokens = [bytes(tf.parts[i]) for i in tf.data]
    scores = [float(sf.parts[i][0]) for i in sf.data] if sf else [0.0] * len(tokens)

    def get_scalar(key, default=0):
        f = reader.get_field(key)
        return int(f.parts[-1][0]) if f else default

    n_vocab = len(tokens)
    n_embd = get_scalar("llama.embedding_length", 288)
    n_head = get_scalar("llama.attention.head_count", 6)
    n_layer = get_scalar("llama.block_count", 6)
    n_rot = get_scalar("llama.rope.dimension_count", 48)
    ftype = get_scalar("general.file_type", 2)
    n_mult = 256

    with open(out_ggjt_path, "wb") as f:
        # Magic & version
        f.write(struct.pack("<II", 0x67676a74, version))
        # Hparams
        f.write(struct.pack("<7i", n_vocab, n_embd, n_mult, n_head, n_layer, n_rot, ftype))

        # Vocab
        for tok, score in zip(tokens, scores):
            f.write(struct.pack("<I", len(tok)))
            f.write(tok)
            f.write(struct.pack("<f", score))

        # Tensors
        for tensor in reader.tensors:
            name = map_to_ggjt_name(tensor.name)
            name_bytes = name.encode("utf-8")
            n_dims = len(tensor.shape)
            ggml_type = int(tensor.tensor_type)

            f.write(struct.pack("<iii", n_dims, len(name_bytes), ggml_type))
            ne = [int(d) for d in tensor.shape]
            f.write(struct.pack(f"<{n_dims}i", *ne))
            f.write(name_bytes)

            # Pad to 32 bytes
            pad = (32 - (f.tell() % 32)) % 32
            f.write(b"\x00" * pad)

            raw_data = bytes(tensor.data)

            if version == 3:
                f.write(raw_data)
            elif version == 2:
                # Convert Q4_0: fp16 scale -> float32 scale
                if ggml_type == 2: # GGML_TYPE_Q4_0
                    num_blocks = len(raw_data) // 18
                    v2_data = bytearray(num_blocks * 20)
                    for b in range(num_blocks):
                        d_fp16 = np.frombuffer(raw_data[b*18 : b*18 + 2], dtype=np.float16)[0]
                        d_f32 = float(d_fp16)
                        struct.pack_into("<f", v2_data, b*20, d_f32)
                        v2_data[b*20 + 4 : b*20 + 20] = raw_data[b*18 + 2 : b*18 + 18]
                    f.write(v2_data)
                elif ggml_type == 8: # GGML_TYPE_Q8_0
                    num_blocks = len(raw_data) // 34
                    v2_data = bytearray(num_blocks * 36)
                    for b in range(num_blocks):
                        d_fp16 = np.frombuffer(raw_data[b*34 : b*34 + 2], dtype=np.float16)[0]
                        d_f32 = float(d_fp16)
                        struct.pack_into("<f", v2_data, b*36, d_f32)
                        v2_data[b*36 + 4 : b*36 + 36] = raw_data[b*34 + 2 : b*34 + 34]
                    f.write(v2_data)
                else:
                    f.write(raw_data)
            elif version == 1:
                # Convert Q4_0: fp16 scale -> float32 scale AND swap nibbles
                if ggml_type == 2:
                    num_blocks = len(raw_data) // 18
                    v1_data = bytearray(num_blocks * 20)
                    for b in range(num_blocks):
                        d_fp16 = np.frombuffer(raw_data[b*18 : b*18 + 2], dtype=np.float16)[0]
                        d_f32 = float(d_fp16)
                        struct.pack_into("<f", v1_data, b*20, d_f32)
                        qs_in = raw_data[b*18 + 2 : b*18 + 18]
                        for j in range(16):
                            x = qs_in[j]
                            v1_data[b*20 + 4 + j] = ((x >> 4) & 0x0F) | ((x & 0x0F) << 4)
                    f.write(v1_data)
                elif ggml_type == 8:
                    num_blocks = len(raw_data) // 34
                    v1_data = bytearray(num_blocks * 36)
                    for b in range(num_blocks):
                        d_fp16 = np.frombuffer(raw_data[b*34 : b*34 + 2], dtype=np.float16)[0]
                        d_f32 = float(d_fp16)
                        struct.pack_into("<f", v1_data, b*36, d_f32)
                        v1_data[b*36 + 4 : b*36 + 36] = raw_data[b*34 + 2 : b*34 + 34]
                    f.write(v1_data)
                else:
                    f.write(raw_data)

    print(f"Synthesized GGJT v{version}: {out_ggjt_path} ({os.path.getsize(out_ggjt_path)} bytes)")

def test_conversion():
    os.makedirs("build/test_convert", exist_ok=True)

    # Baseline output from source GGUF
    res_orig = subprocess.run([CLI_BIN, "-m", MODEL_SRC, "-p", "Once upon a time", "-n", "16", "--temp", "0", "--render", "final"], capture_output=True, text=True)
    assert res_orig.returncode == 0, f"Original CLI failed: {res_orig.stderr}"
    baseline_text = res_orig.stdout.strip()

    # 1. Test GGJT v3
    v3_ggjt = "build/test_convert/model_v3.ggjt"
    v3_gguf = "build/test_convert/roundtrip_v3.gguf"
    make_ggjt(MODEL_SRC, v3_ggjt, version=3)

    res = subprocess.run([CONVERT_BIN, v3_ggjt, v3_gguf, "--ctx", "128", "--eps", "1e-5"], capture_output=True, text=True)
    assert res.returncode == 0, f"Convert v3 failed: {res.stderr}\n{res.stdout}"

    res_rt = subprocess.run([CLI_BIN, "-m", v3_gguf, "-p", "Once upon a time", "-n", "16", "--temp", "0", "--render", "final"], capture_output=True, text=True)
    assert res_rt.returncode == 0, f"Inference on v3 roundtrip failed: {res_rt.stderr}"
    assert baseline_text == res_rt.stdout.strip(), f"Mismatch v3: {repr(baseline_text)} vs {repr(res_rt.stdout.strip())}"
    print(f"Test 1 (GGJT v3 roundtrip exact parity): Passed! Output: {repr(baseline_text)}")

    # 2. Test GGJT v2
    v2_ggjt = "build/test_convert/model_v2.ggjt"
    v2_gguf = "build/test_convert/roundtrip_v2.gguf"
    make_ggjt(MODEL_SRC, v2_ggjt, version=2)

    res = subprocess.run([CONVERT_BIN, v2_ggjt, v2_gguf, "--ctx", "128", "--eps", "1e-5"], capture_output=True, text=True)
    assert res.returncode == 0, f"Convert v2 failed: {res.stderr}\n{res.stdout}"

    res_v2 = subprocess.run([CLI_BIN, "-m", v2_gguf, "-p", "Once upon a time", "-n", "16", "--temp", "0", "--render", "final"], capture_output=True, text=True)
    assert res_v2.returncode == 0, f"Inference on v2 roundtrip failed: {res_v2.stderr}"
    assert baseline_text == res_v2.stdout.strip(), f"Mismatch v2: {repr(baseline_text)} vs {repr(res_v2.stdout.strip())}"
    print(f"Test 2 (GGJT v2 repack exact parity): Passed! Output: {repr(baseline_text)}")

    # 3. Test GGJT v1
    v1_ggjt = "build/test_convert/model_v1.ggjt"
    v1_gguf = "build/test_convert/roundtrip_v1.gguf"
    make_ggjt(MODEL_SRC, v1_ggjt, version=1)

    res = subprocess.run([CONVERT_BIN, v1_ggjt, v1_gguf, "--ctx", "128", "--eps", "1e-5"], capture_output=True, text=True)
    assert res.returncode == 0, f"Convert v1 failed: {res.stderr}\n{res.stdout}"

    res_v1 = subprocess.run([CLI_BIN, "-m", v1_gguf, "-p", "Once upon a time", "-n", "16", "--temp", "0", "--render", "final"], capture_output=True, text=True)
    assert res_v1.returncode == 0, f"Inference on v1 roundtrip failed: {res_v1.stderr}"
    assert baseline_text == res_v1.stdout.strip(), f"Mismatch v1: {repr(baseline_text)} vs {repr(res_v1.stdout.strip())}"
    print(f"Test 3 (GGJT v1 unshuffle & repack exact parity): Passed! Output: {repr(baseline_text)}")

    print("\nAll diffusion-convert GGJT v1, v2, v3 tests PASSED with 100% exact parity!")

if __name__ == "__main__":
    test_conversion()
