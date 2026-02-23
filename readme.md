项目简介  本地文档问答系统（Persona Assistant API / RAG 小项目）

这是一个基于 FastAPI 的“资料问答型”RAG 本地文档问答系统系统服务。用户可以上传个人资料（如 txt / md ），系统会对资料进行切分、向量化与检索，然后基于检索到的证据调用本地大模型生成回答，并返回引用来源，尽量避免编造。 

核心能力
1、文件上传与管理
支持上传文件到本地 data/raw，支持查看已上传文件列表。

2、索引构建（RAG 数据准备）
从上传的文件中抽取文本（支持 .txt/.md/），对文本做脱敏处理，将文档按“项目块/标题块优先、滑窗兜底”的策略进行切分（chunking），使用 Ollama embedding 生成向量，并写入 Qdrant 向量库，同时构建 BM25 索引并保存到本地，用于混合检索 。

3、混合检索 + 精排
检索阶段结合：向量检索（Qdrant）+ BM25（本地）并用 RRF 融合排序，对候选 chunk 进行 rerank 精排，提高相关性 。

4、问答接口（带引用）
POST /chat：对用户问题进行路由，检索证据后调用本地模型生成回答，返回结构包含 answer 和 citations，对寒暄/“你是谁、怎么用”等问题有内置回复逻辑 。

5、Profile 信息抽取
会从上传资料中尝试抽取个人字段，Profile 保存为 data/profile.json，提供 GET /profile 与 POST /profile/rebuild 接口 。

后端启动
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000


前端启动npm run dev
