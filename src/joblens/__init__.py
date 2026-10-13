"""职探 JobLens: 把一份岗位 JD 和你的简历做深度对比, 输出匹配度、改写建议和面试重点。

包结构速览:
- graph.py      LangGraph 主图(fetch_jd → extract → analyze → advise)
- state.py      AgentState 与三个 Pydantic 结构化输出 schema
- config.py     环境变量配置(LLM / embedding / 数据路径 / CORS)
- memory.py     LangGraph checkpointer 持久化记忆(thread_id 多轮续跑)
- cli.py        命令行入口
- nodes/        四个图节点的实现
- prompts/      发给 LLM 的中文 prompt 模板
- rag/          embedding + Chroma 向量库 + 相似历史检索
- api/          FastAPI 服务(REST + SSE + 静态前端)
- history/      投递历史的 SQLite 存档
- tools/        PDF 解析与网页抓取
- web/static/   前端三件套(原生 HTML/JS/CSS, 无构建步骤)
"""
