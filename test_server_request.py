import json
import urllib.request
import time

tools = []
for i in range(12):
    tools.append({
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Tool description for tool {i} with several details and instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument description"},
                    "arg2": {"type": "integer", "description": "Second argument count"}
                },
                "required": ["arg1"]
            }
        }
    })

prompt_text = "This is a long test prompt sentence with lots of details about programming languages and system design. " * 200
payload = {
    "model": "gemma",
    "messages": [{"role": "user", "content": prompt_text}],
    "tools": tools,
    "stream": True,
    "max_completion_tokens": 50,
    "reasoning_effort": "low",
    "stream_options": {"include_usage": True}
}

req = urllib.request.Request(
    "http://127.0.0.1:4096/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print(f"Payload size: {len(json.dumps(payload))} bytes, prompt text: {len(prompt_text)} chars")
try:
    with urllib.request.urlopen(req, timeout=30) as response:
        for line in response:
            print(line.decode("utf-8", errors="replace"), end="")
except Exception as e:
    print("Error:", e)

