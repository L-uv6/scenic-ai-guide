"""
Embedding 模型选型对比测试
============================
候选模型：
  1. text2vec-base-chinese   (shibing624, 本地, 768维)
  2. BGE-M3                  (BAAI, 本地, 1024维)
  3. OpenAI text-embedding-3-small  (API, 1536维, 可选)

评估维度：检索命中率、语义理解、推理速度
"""

import json
import time
import os
import sys
import signal
from typing import List, Dict, Tuple
from contextlib import contextmanager

import chromadb

# ============================================================
# 1. 测试数据
# ============================================================

TEST_CORPUS = [
    # venue
    {"id": "venue_001", "type": "venue",
     "content": "广州鲁迅纪念馆位于广州市越秀区文明路215号，馆址所在地为原中山大学钟楼，这里是1927年鲁迅在广州居住和任教的地方。纪念馆占地面积约1900平方米，设有鲁迅生平陈列展、鲁迅在广州专题展等。"},
    {"id": "venue_002", "type": "venue",
     "content": "钟楼建于1905年，是广州近代重要历史建筑，楼高四层，砖木结构，中西合璧风格。1927年鲁迅担任中山大学文学系主任时，就住在钟楼二层。"},
    {"id": "venue_003", "type": "venue",
     "content": "纪念馆开放时间为周二至周日9:00-17:00，周一闭馆。免费参观，需凭有效证件领取参观券。地址位于广州市越秀区文明路，地铁1号线公园前站可达。"},
    # work
    {"id": "work_001", "type": "work",
     "content": "我翻开历史一查，这历史没有年代，歪歪斜斜的每页上都写着'仁义道德'几个字。我横竖睡不着，仔细看了半夜，才从字缝里看出字来，满本都写着两个字是'吃人'！"},
    {"id": "work_002", "type": "work",
     "content": "我在朦胧中，眼前展开一片海边碧绿的沙地来，上面深蓝的天空中挂着一轮金黄的圆月。我想：希望是本无所谓有，无所谓无的。这正如地上的路；其实地上本没有路，走的人多了，也便成了路。"},
    {"id": "work_003", "type": "work",
     "content": "横眉冷对千夫指，俯首甘为孺子牛。躲进小楼成一统，管他冬夏与春秋。"},
    {"id": "work_004", "type": "work",
     "content": "真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。这是怎样的哀痛者和幸福者？然而造化又常常为庸人设计，以时间的流驶，来洗涤旧迹，仅使留下淡红的血色和微漠的悲哀。"},
    {"id": "work_005", "type": "work",
     "content": "在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。"},
    # bio
    {"id": "bio_001", "type": "bio",
     "content": "1927年1月18日，鲁迅从厦门乘船抵达广州，应中山大学之聘任文学系主任兼教务主任。鲁迅住进中山大学钟楼二楼，即今日广州鲁迅纪念馆所在地。"},
    {"id": "bio_002", "type": "bio",
     "content": "1881年9月25日，鲁迅出生于浙江绍兴府会稽县东昌坊口新台门周家。原名周樟寿，后改名周树人，字豫才。"},
    {"id": "bio_003", "type": "bio",
     "content": "1902年鲁迅赴日本留学，先入东京弘文学院学习日语，1904年入仙台医学专门学校学医。1906年在校观看日俄战争幻灯片，看到中国人围观同胞被处刑却面无表情，深受刺激，决定弃医从文。"},
    # quote
    {"id": "quote_001", "type": "quote",
     "content": "不在沉默中爆发，就在沉默中灭亡。"},
    {"id": "quote_002", "type": "quote",
     "content": "哀其不幸，怒其不争。"},
    {"id": "quote_003", "type": "quote",
     "content": "我们自古以来，就有埋头苦干的人，有拼命硬干的人，有为民请命的人，有舍身求法的人……这就是中国的脊梁。"},
    # persona
    {"id": "persona_001", "type": "persona",
     "content": "鲁迅的文风以犀利、冷峻、深刻著称于世。他善用反讽、白描和寓言手法，语言简练而有力，常常在不动声色的叙述中蕴含着强烈的情感。"},
]

TEST_QUERIES = [
    ("广州鲁迅纪念馆在哪里？", "venue_001"),
    ("钟楼是什么时候建的？", "venue_002"),
    ("纪念馆的开放时间", "venue_003"),
    ("吃人的礼教", "work_001"),
    ("地上本没有路", "work_002"),
    ("横眉冷对千夫指", "work_003"),
    ("真正的猛士", "work_004"),
    ("一株是枣树", "work_005"),
    ("鲁迅什么时候到广州的？", "bio_001"),
    ("鲁迅出生在哪里？", "bio_002"),
    ("鲁迅为什么弃医从文？", "bio_003"),
    ("不在沉默中爆发", "quote_001"),
    ("哀其不幸", "quote_002"),
    ("中国的脊梁是什么意思", "quote_003"),
    ("鲁迅的写作风格", "persona_001"),
    ("鲁迅在广州做了什么？", "bio_001"),
    ("鲁迅对青年有什么期望？", "work_002"),
]


# ============================================================
# 2. 工具函数
# ============================================================

@contextmanager
def timeout_ctx(seconds: int, label: str):
    """跨平台超时上下文管理器"""
    if sys.platform == "win32":
        # Windows: 启动一个 watchdog 线程
        import threading
        timed_out = [False]

        def watchdog():
            timed_out[0] = True
            # 无法在 Windows 上优雅终止，只能标记

        timer = threading.Timer(seconds, watchdog)
        timer.start()
        try:
            yield timed_out
        finally:
            timer.cancel()
    else:
        signal.signal(signal.SIGALRM, lambda sig, frame: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


def load_model_safe(model_name: str, label: str, timeout_s: int = 300):
    """安全加载模型，带超时"""
    from chromadb.utils import embedding_functions

    print(f"  加载 {label} ({model_name}) ...", flush=True)
    t0 = time.time()

    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        # 预热
        t1 = time.time()
        print(f"    下载/加载: {t1-t0:.1f}s, 预热推理中...", flush=True)
        _ = ef(["测试文本"])
        t2 = time.time()
        print(f"    ✓ 就绪 (总计 {t2-t0:.1f}s, 推理 {t2-t1:.1f}s)", flush=True)
        return ef, {"load_time": t2-t0, "inference_time": t2-t1}
    except Exception as e:
        print(f"    ✗ 失败: {e}", flush=True)
        return None, None


def evaluate_model(name: str, ef, corpus: List[Dict], queries: List[Tuple[str, str]]) -> Dict:
    """检索评估"""
    print(f"  评测中...", flush=True)

    client = chromadb.Client()
    try:
        client.delete_collection("eval_temp")
    except:
        pass
    collection = client.create_collection(
        name="eval_temp",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 入库
    ids = [d["id"] for d in corpus]
    documents = [d["content"] for d in corpus]
    metadatas = [{"type": d["type"]} for d in corpus]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # 查询
    results = []
    inference_times = []

    for query, expected_id in queries:
        t0 = time.time()
        r = collection.query(query_texts=[query], n_results=5)
        t1 = time.time()
        inference_times.append(t1 - t0)

        hit_ids = r["ids"][0]
        hit_distances = r["distances"][0]

        results.append({
            "query": query,
            "expected": expected_id,
            "top1": hit_ids[0] == expected_id,
            "top3": expected_id in hit_ids[:3],
            "top5": expected_id in hit_ids[:5],
            "hits": hit_ids[:5],
            "top1_distance": hit_distances[0],
        })

    client.delete_collection("eval_temp")

    top1_correct = sum(1 for r in results if r["top1"])
    top3_correct = sum(1 for r in results if r["top3"])
    top5_correct = sum(1 for r in results if r["top5"])
    total = len(queries)
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

    return {
        "name": name,
        "total_queries": total,
        "top1": top1_correct, "top1_rate": top1_correct / total,
        "top3": top3_correct, "top3_rate": top3_correct / total,
        "top5": top5_correct, "top5_rate": top5_correct / total,
        "avg_inference_ms": avg_time * 1000,
        "details": results,
    }


# ============================================================
# 3. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Embedding 模型选型对比测试")
    print("  广州鲁迅纪念馆 × 鲁迅数字人")
    print("=" * 60)
    print(f"  语料: {len(TEST_CORPUS)} 条 | 查询: {len(TEST_QUERIES)} 条")
    print()

    all_results = []

    # ---- 候选中文 Embedding 模型列表 ----
    CANDIDATES = [
        ("shibing624/text2vec-base-chinese", "text2vec-base-chinese"),
        ("BAAI/bge-small-zh-v1.5",    "bge-small-zh-v1.5"),
        ("BAAI/bge-base-zh-v1.5",     "bge-base-zh-v1.5"),
        ("BAAI/bge-large-zh-v1.5",    "bge-large-zh-v1.5"),
    ]

    for model_path, label in CANDIDATES:
        ef, info = load_model_safe(model_path, label)
        if ef:
            result = evaluate_model(label, ef, TEST_CORPUS, TEST_QUERIES)
            result["load_info"] = info
            all_results.append(result)
            print(f"    Top-1: {result['top1_rate']:.0%}  Top-3: {result['top3_rate']:.0%}  "
                  f"Top-5: {result['top5_rate']:.0%}  {result['avg_inference_ms']:.0f}ms/query", flush=True)
            print()

    # ---- OpenAI API ----
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from chromadb.utils import embedding_functions
        print("[3/3] 评测 OpenAI text-embedding-3-small ...")
        try:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, model_name="text-embedding-3-small"
            )
            result = evaluate_model("openai-embedding-3-small", openai_ef, TEST_CORPUS, TEST_QUERIES)
            all_results.append(result)
            print(f"    Top-1: {result['top1_rate']:.0%}  Top-3: {result['top3_rate']:.0%}  "
                  f"Top-5: {result['top5_rate']:.0%}  {result['avg_inference_ms']:.0f}ms/query", flush=True)
        except Exception as e:
            print(f"    ✗ 失败: {e}", flush=True)

    if not all_results:
        print("❌ 无可用模型！")
        return

    # ---- 对比表 ----
    print()
    print("=" * 70)
    print("  对比总表")
    print("=" * 70)
    print(f"{'模型':<30} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'延迟':>8} {'加载':>8}")
    print("-" * 70)
    for r in all_results:
        load_s = r.get("load_info", {}).get("load_time", 0)
        print(f"{r['name']:<30} {r['top1_rate']:>5.0%} {r['top3_rate']:>5.0%} "
              f"{r['top5_rate']:>5.0%} {r['avg_inference_ms']:>5.0f}ms {load_s:>5.0f}s")

    # ---- 失败查询 ----
    for r in all_results:
        failed = [d for d in r["details"] if not d["top5"]]
        if failed:
            print(f"\n--- {r['name']} Top-5 未命中 ---")
            for f in failed:
                print(f"  '{f['query']}' → 期望:{f['expected']} 实际:{f['hits']}")

    # ---- 推荐 ----
    print()
    print("=" * 70)
    print("  结论")
    print("=" * 70)
    best = max(all_results, key=lambda r: r["top3_rate"])
    print(f"  ★ 推荐: {best['name']}")
    print(f"    Top-3 命中率: {best['top3_rate']:.0%}")
    print(f"    推理延迟: {best['avg_inference_ms']:.0f}ms/query")
    for r in all_results:
        if r != best:
            print(f"    vs {r['name']}: Top-3 高 {best['top3_rate']-r['top3_rate']:.0%}, "
                  f"延迟{'低' if best['avg_inference_ms']<r['avg_inference_ms'] else '高'}"
                  f"{abs(best['avg_inference_ms']-r['avg_inference_ms']):.0f}ms")

    # ---- 保存 ----
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "docs", "embedding-eval-results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  详细结果: {out_path}")


if __name__ == "__main__":
    main()
