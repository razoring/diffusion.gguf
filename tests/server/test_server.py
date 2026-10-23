import subprocess
import time
import requests
import json
import sys
import os

PORT = 8097
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_PATH = "models/stories15M-q4_0.gguf"
SERVER_BIN = os.path.abspath("build/bin/Release/diffusion-server.exe")

def run_server_tests():
    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "-np", "2",
        "--api-key", "test-secret-key",
        "--mode", "block",
        "--block", "8"
    ]

    print(f"Starting diffusion-server on port {PORT}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # Wait for server to come up (up to 15s)
        for _ in range(60):
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("Server failed to start within timeout")

        print("Server is up and healthy.")

        headers = {
            "Authorization": "Bearer test-secret-key",
            "Content-Type": "application/json"
        }

        # 1. Test Auth failure
        r = requests.get(f"{BASE_URL}/v1/models")
        assert r.status_code == 401, f"Expected 401 without auth, got {r.status_code}"
        print("Test 1 (Auth failure check): Passed")

        # 2. Test /v1/models with auth
        r = requests.get(f"{BASE_URL}/v1/models", headers=headers)
        assert r.status_code == 200
        models_data = r.json()
        assert models_data["object"] == "list"
        assert len(models_data["data"]) >= 1
        print("Test 2 (/v1/models): Passed")

        # 3. Test /v1/completions non-stream
        payload = {
            "prompt": "Once upon a time",
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False
        }
        r = requests.post(f"{BASE_URL}/v1/completions", headers=headers, json=payload)
        assert r.status_code == 200, f"Got status {r.status_code}: {r.text}"
        data = r.json()
        assert "choices" in data and len(data["choices"]) > 0
        non_stream_text = data["choices"][0]["text"]
        assert len(non_stream_text) > 0
        assert "diffusion_iterations" in data["usage"]
        print(f"Test 3 (/v1/completions non-stream): Passed. Generated: {repr(non_stream_text)}")

        # 4. Test /v1/completions stream (SSE)
        payload["stream"] = True
        r = requests.post(f"{BASE_URL}/v1/completions", headers=headers, json=payload, stream=True)
        assert r.status_code == 200

        streamed_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                delta = chunk["choices"][0].get("text", "")
                streamed_text += delta

        print(f"Test 4 (/v1/completions stream): Passed. Streamed: {repr(streamed_text)}")
        assert streamed_text == non_stream_text, f"Streamed mismatch: {repr(streamed_text)} vs {repr(non_stream_text)}"

        # 5. Test /v1/chat/completions non-stream
        chat_payload = {
            "messages": [{"role": "user", "content": "Once upon a time"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False
        }
        r = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=chat_payload)
        assert r.status_code == 200
        chat_data = r.json()
        assert "choices" in chat_data and len(chat_data["choices"]) > 0
        chat_non_stream_text = chat_data["choices"][0]["message"]["content"]
        assert len(chat_non_stream_text) > 0
        print(f"Test 5 (/v1/chat/completions non-stream): Passed. Response: {repr(chat_non_stream_text)}")

        # 6. Test /v1/chat/completions stream (SSE)
        chat_payload["stream"] = True
        r = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=chat_payload, stream=True)
        assert r.status_code == 200

        chat_streamed_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                chat_streamed_text += delta

        print(f"Test 6 (/v1/chat/completions stream): Passed. Streamed: {repr(chat_streamed_text)}")
        assert chat_streamed_text == chat_non_stream_text, f"Chat streamed mismatch: {repr(chat_streamed_text)} vs {repr(chat_non_stream_text)}"

        # 7. Test logprobs in /v1/completions
        lp_payload = {
            "prompt": "Hello world",
            "max_tokens": 8,
            "logprobs": True,
            "top_logprobs": 3,
            "stream": False
        }
        r = requests.post(f"{BASE_URL}/v1/completions", headers=headers, json=lp_payload)
        assert r.status_code == 200
        lp_data = r.json()
        choices = lp_data["choices"]
        assert len(choices) > 0 and choices[0]["logprobs"] is not None
        assert "content" in choices[0]["logprobs"]
        assert len(choices[0]["logprobs"]["content"]) > 0
        print("Test 7 (Logprobs and top_logprobs): Passed")

        # 8. Test /slots and /props
        r = requests.get(f"{BASE_URL}/slots", headers=headers)
        assert r.status_code == 200
        slots = r.json()
        assert len(slots) == 2, f"Expected 2 slots, got {len(slots)}"
        print("Test 8 (/slots): Passed")

        r = requests.get(f"{BASE_URL}/props", headers=headers)
        assert r.status_code == 200
        props = r.json()
        assert props["total_slots"] == 2
        print("Test 8b (/props): Passed")

        # 9. Test /metrics (Prometheus)
        r = requests.get(f"{BASE_URL}/metrics", headers=headers)
        assert r.status_code == 200
        assert "diffusion_requests_total" in r.text
        assert "diffusion_slots_total 2" in r.text
        print("Test 9 (/metrics): Passed")

        # 10. Test /tokenize and /detokenize
        r = requests.post(f"{BASE_URL}/tokenize", headers=headers, json={"content": "Once upon a time"})
        assert r.status_code == 200
        toks = r.json()["tokens"]
        assert len(toks) > 0

        r = requests.post(f"{BASE_URL}/detokenize", headers=headers, json={"tokens": toks})
        assert r.status_code == 200
        detok = r.json()["content"]
        assert "Once" in detok
        print("Test 10 (/tokenize and /detokenize): Passed")

        # 11. Test /v1/embeddings
        r = requests.post(f"{BASE_URL}/v1/embeddings", headers=headers, json={"input": "Hello world"})
        assert r.status_code == 200
        emb_data = r.json()
        assert emb_data["object"] == "list"
        assert len(emb_data["data"]) == 1
        assert len(emb_data["data"][0]["embedding"]) > 0
        print("Test 11 (/v1/embeddings): Passed")

        # 12. Test /infill
        r = requests.post(f"{BASE_URL}/infill", headers=headers, json={
            "input_prefix": "def add(a, b):\n    return",
            "input_suffix": "\nprint(add(2, 3))",
            "max_tokens": 8
        })
        assert r.status_code == 200
        infill_data = r.json()
        assert "content" in infill_data
        print("Test 12 (/infill): Passed")

        # 13. Test /lora-adapters
        r = requests.get(f"{BASE_URL}/lora-adapters", headers=headers)
        assert r.status_code == 200
        print("Test 13 (/lora-adapters): Passed")

        print("\nAll diffusion-server API parity tests PASSED successfully!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    run_server_tests()
