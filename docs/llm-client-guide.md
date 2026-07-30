# LLM 调用使用说明

> `scripts/llm_client.py` · 张嘉欣  
> 直接 import 使用，不用管重试、限流、超时。

## 一步调用

```python
from scripts.llm_client import chat

reply = chat(
    system_prompt="你是讲解员，只根据参考资料回答。",
    context="广州鲁迅纪念馆位于文明路215号…",   # RAG 检索到的知识片段
    user_query="纪念馆在哪里？",
)
print(reply)
```

`context` 为空时模型用自身知识回答；非空时会拼成 `【参考资料】+ 【用户问题】` 格式传给模型。

## 多轮对话

```python
from scripts.llm_client import LLMClient

client = LLMClient()

# 每次调用自动保留最近 5 轮历史
reply1 = client.chat_with_history(system, ctx, "鲁迅是谁？")
reply2 = client.chat_with_history(system, ctx, "他为什么弃医从文？")  # 知道"他"指鲁迅

client.clear_history()  # 切换访客时清空
```

## 几个坑

1. `.env` 必须放在 `scenic-ai-guide/` 根目录，否则报 `LLM_API_KEY 未设置`
2. `.env` 不要提交到 Git（已在 `.gitignore` 中）
3. 网络偶发 SSL 错误会自动重试 3 次（1s→2s→4s），一般第二次就通了
4. 换模型直接改 `.env` 中 `LLM_MODEL=glm-4-flash` 即可

## 自测

```bash
cd scenic-ai-guide
python scripts/llm_client.py
```

输出 `OK 自测通过` = 配置正确，可以正常调用。
