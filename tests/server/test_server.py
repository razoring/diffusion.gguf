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

        # 7. Test invalid parameters (n > 1, logprobs)
        r = requests.post(f"{BASE_URL}/v1/completions", headers=headers, json={"prompt": "hi", "n": 2})
        assert r.status_code == 400
        r = requests.post(f"{BASE_URL}/v1/completions", headers=headers, json={"prompt": "hi", "logprobs": 5})
        assert r.status_code == 400
        print("Test 7 (Rejections for n > 1 and logprobs): Passed")

        print("\nAll diffusion-server API tests PASSED successfully!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    run_server_tests()
