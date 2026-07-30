# scripts/llm_test.py
import os
import json
import argparse
import requests
# 在文件开头添加
from dotenv import load_dotenv
load_dotenv()

def call_llm(provider: str, api_key: str, base_url: str, model: str, system: str, user: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 统一OpenAI风格 messages（大多数国产API都支持类似结构）
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": False
    }

    resp = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 尽量兼容常见返回结构
    # 常见：data["choices"][0]["message"]["content"]
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # 兜底：把原始data打印出来定位字段
        raise RuntimeError(f"Unrecognized response format: {data}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "glm"])
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.getenv("LLM_BASE_URL", ""))

    parser.add_argument("--question", type=str, default="鲁迅是谁？用一句话回答。")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key: set env LLM_API_KEY or pass --api_key")
    if not args.base_url:
        raise SystemExit("Missing base_url: set env LLM_BASE_URL or pass --base_url")
    if not args.model:
        raise SystemExit("Missing model: set env LLM_MODEL or pass --model")

    system = "你是一个可靠的问答助手。请直接回答，不要输出多余解释。"

    print(f"[llm_test] provider={args.provider}")
    print(f"[llm_test] model={args.model}")
    print(f"[llm_test] base_url={args.base_url}")

    try:
        out = call_llm(
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system=system,
            user=args.question
        )
        print("\n=== LLM OUTPUT ===")
        print(out)
    except requests.exceptions.RequestException as e:
        print(f"\n[llm_test] HTTP/Network error: {e}")
        raise
    except Exception as e:
        print(f"\n[llm_test] Other error: {e}")
        raise


if __name__ == "__main__":
    main()
