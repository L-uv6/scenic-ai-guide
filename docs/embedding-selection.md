# Embedding 模型选型结论

## 一、候选模型

| # | 模型 | 来源 | 维度 | 大小 | 方式 |
|---|------|------|------|------|------|
| 1 | `text2vec-base-chinese` | shibing624 | 768 | ~400MB | 本地 |
| 2 | `bge-small-zh-v1.5` | BAAI | 512 | ~100MB | 本地 |
| 3 | `bge-base-zh-v1.5` | BAAI | 768 | ~400MB | 本地 |

---

## 二、测试结果对比

| 模型 | Top-1 | Top-3 | Top-5 | 延迟 | 加载 | 大小 |
|------|-------|-------|-------|------|------|------|
| text2vec-base-chinese | 88% | 94% | 94% | 46ms | 27s | ~400MB |
| **bge-small-zh-v1.5** | **94%** | **94%** | **94%** | **14ms** | **76s** | **~100MB** |
| bge-base-zh-v1.5 | 94% | 94% | 94% | 49ms | 112s | ~400MB |


---

## 三、结论

### ★ 选用：`BAAI/bge-small-zh-v1.5`

---

## 四、技术栈确定

```
Embedding 模型: BAAI/bge-small-zh-v1.5
向量数据库:     ChromaDB
加载方式:       SentenceTransformerEmbeddingFunction
推理延迟:       ~14ms/query
模型大小:       ~100MB
```
建议采用“Embedding (语义检索) + Reranker (重排序模型)”机制
---
