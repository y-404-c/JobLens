"""FastAPI 服务包: server.py 是应用入口, routes.py 定义接口, schemas.py 定义响应模型。

启动: uv run uvicorn joblens.api.server:app --reload
接口: /api/analyze(一次性) / /api/analyze/stream(SSE 流式) / /api/thread/{id}(会话 state)
      /api/history(投递历史 CRUD) / /health(健康检查)
"""
