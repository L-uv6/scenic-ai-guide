"""
窦一禾分工数据生成脚本
生成: venue.json (30条) + bio.json (50条含15条1927广州)
"""
import json, os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)


def call_llm(system_prompt, user_prompt, temperature=0.7):
    """调用LLM，自动处理截断"""
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=16384,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            # 尝试解析
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # 截断修复: 找最后一个完整的 }, 闭合数组
                last_obj = raw.rfind('},')
                if last_obj > 0:
                    fixed = raw[:last_obj + 1] + '\n]'
                else:
                    last_obj = raw.rfind('}')
                    if last_obj > 0:
                        fixed = raw[:last_obj + 1] + '\n]'
                    else:
                        if attempt == 0:
                            continue
                        raise
                return json.loads(fixed)
        except Exception as e:
            if attempt == 0:
                print(f"  重试中... ({e})")
                continue
            raise
    return []


def fix_entry(entry, domain_type, idx):
    """补全/修正单条数据"""
    entry["id"] = f"{domain_type}_{idx:03d}"
    entry["type"] = domain_type
    entry.setdefault("title", "")
    entry.setdefault("content", "")
    entry.setdefault("source", "AI辅助生成，需人工核验")
    entry.setdefault("year", 0)
    entry.setdefault("tags", [])
    entry.setdefault("character_voice", domain_type == "bio" and False)
    entry.setdefault("venue_relevant", False)
    return entry


def generate_venue():
    """生成30条场馆知识"""
    print("=" * 50)
    print("生成 venue.json (30条场馆知识)...")

    prompt = """你是广州鲁迅纪念馆研究员。请生成30条场馆知识JSON数组，严格按以下Schema：

{
  "id": "venue_NNN",
  "type": "venue",
  "title": "标题(10-30字)",
  "content": "详细介绍(100-500字)",
  "source": "广州鲁迅纪念馆/百度百科/越秀区文旅局等",
  "year": 年份或0,
  "tags": ["标签1","标签2","标签3"],
  "character_voice": false,
  "venue_relevant": true
}

必须覆盖：
1. 纪念馆概况(3条): 钟楼建筑历史(1905年建/全国重点文保)、文明路215号位置、隶属广东省博物馆
2. 常设展览"在钟楼上——鲁迅与广东"(5条): 各展区介绍、展览主题、展陈方式
3. 重点展品(6条): 鲁迅手稿复制件、生前用品(金不换毛笔等)、1927广州时期文物、许广平捐赠物品
4. 钟楼建筑(3条): 1905年建、国民党一大旧址(1924)、中西合璧风格、钢筋混凝土结构
5. 白云楼鲁迅故居(3条): 白云路、1927.9迁入、《朝花夕拾》《而已集》编订地、与许广平同居
6. 中山大学关联(3条): 中大钟楼时期(1924-1927)、鲁迅任文学系主任兼教务主任、校史关联
7. 参观信息(3条): 开放时间(9:00-17:00周一闭馆)、免费参观需预约、交通路线(地铁农讲所站)
8. 其他相关(2条): 高第街许地(许广平故居)、广州鲁迅纪念公园(越秀区)
9. 历史背景(2条): 1924国民党一大、1927广州革命策源地

content必须100-500字，信息丰富准确。source要具体。venue_relevant全部true。直接输出纯JSON数组。"""

    data = call_llm("你是精确的JSON生成器，只输出有效JSON数组。", prompt)

    for i, item in enumerate(data, 1):
        fix_entry(item, "venue", i)

    path = os.path.join(OUT_DIR, "venue.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ venue.json: {len(data)} 条 → {path}")
    return data


def generate_bio():
    """生成50条生平知识，分2批"""
    print("=" * 50)
    print("生成 bio.json (50条生平知识)...")

    all_data = []

    # 第1批: 青少年到北京时期 (1881-1926, 20条)
    prompt1 = """你是鲁迅生平研究专家。请生成25条鲁迅生平知识JSON数组(1881-1926年)，严格按Schema：

{
  "id": "bio_NNN",
  "type": "bio",
  "title": "年份+事件概述",
  "content": "时间+地点+人物+事件+意义(100-500字)",
  "source": "《鲁迅年谱》/《鲁迅传》具体出处",
  "year": 事件年份(必填),
  "tags": ["标签"],
  "character_voice": false,
  "venue_relevant": false
}

覆盖：
【青少年(1881-1902)，6条】1881绍兴出生(周樟寿/豫才)、1893祖父周介孚科场案·家道中落、三味书屋·寿镜吾先生、1898南京江南水师学堂→矿路学堂、接触《天演论》新学、1898戊戌变法影响

【留学日本(1902-1909)，5条】1902赴日·弘文学院、1904仙台医专·藤野先生、幻灯片事件·弃医从文(1906)、回东京从事文艺运动·《新生》杂志流产、与周作人合译《域外小说集》

【归国初期(1909-1918)，4条】1909回国任浙江两级师范学堂教员、1912蔡元培邀请任职教育部·迁北京、S会馆钞古碑(1912-1917)、1918《狂人日记》发表于《新青年》(文学革命)

【北京时期(1918-1926)，10条】北大兼课·讲中国小说史、1920《阿Q正传》连载、1923《呐喊》出版、1923兄弟失和·搬出八道湾、1924迁居西三条胡同、《语丝》创刊(1924)、女师大风潮(1925)、1926《彷徨》出版、三一八惨案·《记念刘和珍君》、1926离京避难

year必须准确。直接输出纯JSON数组。"""

    data1 = call_llm("你是精确的JSON生成器。", prompt1)
    all_data.extend(data1)
    print(f"  [1/2] 生成 {len(data1)} 条")

    # 第2批: 厦门→广州→上海 (1926-1936, 25条，重点1927广州!)
    prompt2 = """你是鲁迅生平研究专家。请生成25条鲁迅生平知识JSON数组(1926-1936年)，严格按Schema：

{
  "id": "bio_NNN",
  "type": "bio",
  "title": "年份+事件概述",
  "content": "时间+地点+人物+事件+意义(100-500字)",
  "source": "《鲁迅年谱》/《鲁迅传》具体出处",
  "year": 事件年份(必填),
  "tags": ["标签"],
  "character_voice": false,
  "venue_relevant": 与广州/纪念馆相关为true
}

覆盖：
【厦门(1926.9-1927.1)，3条】1926.9赴厦门大学任国文系教授、厦大期间编写《中国文学史略》、1927.1离开厦门赴广州

【广州时期(1927.1-1927.9)，重点!共15条全部venue_relevant=true,tags加"广州"】
1927.1.18抵达广州(厦门乘船至广州黄埔港)、入住文明路中山大学钟楼二楼、任中大文学系主任兼教务主任(聘书/待遇/权力)、《读书与革命》演讲(1927.3.1开学典礼)、广州四一五事变(1927.4.15国民党清党·鲁迅营救被捕学生)、辞去中大一切职务(1927.4.21致信朱家骅)、1927.5-9隐居白云楼编订《朝花夕拾》、编订《而已集》(1927广州杂文·答有恒先生等)、广州期间8次重要演讲(中大/岭南大学/知用中学)、与许广平关系在广州确立(从师生到伴侣)、鲁迅广州时期思想转变(进化论→阶级论)、广州时期交友(许寿裳/孙伏园/傅斯年/顾颉刚)、创作《小约翰》译稿(1927.5-6)、1927.9.27离开广州赴上海、与广东籍革命青年交往(毕磊等中共党员)

【上海时期(1927.10-1936)，7条】
1927.10.3抵达上海定居景云里、1928-1929革命文学论争、1930左联成立任盟主、1932内山书店避居·柔石等五烈士、《申报·自由谈》专栏(1933)、1934-1935扶持萧红萧军等青年作家、1936.10.19病逝上海(遗嘱·"让他们怨恨去，我也一个都不宽恕")

year必须准确。广州15条venue_relevant=true。直接输出纯JSON数组。"""

    data2 = call_llm("你是精确的JSON生成器。", prompt2)
    all_data.extend(data2)
    print(f"  [2/2] 生成 {len(data2)} 条")

    # 统一处理
    for i, item in enumerate(all_data, 1):
        fix_entry(item, "bio", i)
        # 修正venue_relevant: 1927年+广州关键词标记true
        year = item.get("year", 0)
        content = item.get("content", "") + "".join(item.get("tags", []))
        if year == 1927 and any(kw in content for kw in ["广州", "钟楼", "白云楼", "四一五", "中大", "中山大学", "而已集"]):
            item["venue_relevant"] = True

    path = os.path.join(OUT_DIR, "bio.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    gz_count = sum(1 for e in all_data if e.get("venue_relevant"))
    print(f"  ✅ bio.json: {len(all_data)} 条 (其中venue_relevant={gz_count}条) → {path}")
    return all_data


if __name__ == "__main__":
    venue = generate_venue()
    bio = generate_bio()

    print(f"\n{'='*50}")
    print("窦一禾分工数据生成完毕:")
    print(f"  venue.json: {len(venue)} 条 (目标≥30)")

    gz = sum(1 for e in bio if e.get("venue_relevant"))
    print(f"  bio.json:   {len(bio)} 条 (目标≥50, 广州相关{gz}条≥15)")
    print(f"  输出目录: {OUT_DIR}")
