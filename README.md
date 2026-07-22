# 景点知识库驱动的讲解AI

基于大语言模型（LLM）和RAG技术的智能景点讲解系统。用户提问景点相关问题，AI像真人导游一样给出准确、生动的讲解。

## 项目阶段

1. **知识库构建** — 收集景点资料，向量化存储，构建可查询的知识库
2. **对话管线开发** — 基于RAG的问答流程，可控叙事生成
3. **鲁棒性优化** — 幻觉检测、一致性评估、对话中断恢复
4. **端侧部署** — 模型蒸馏、量化压缩、Jetson边缘设备部署

## 技术栈

（启动会后确定）

## 团队


## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd scenic-ai-guide

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置API Key
cp .env.example .env
# 编辑 .env 填入你的API Key
```
