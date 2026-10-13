"""LangGraph 持久化记忆(checkpointer)的封装。

JD 岗位常见技术栈里所说的"LangGraph checkpoint / thread_id 持久化记忆", 落点就在这一层:

- 存储后端挑的是 SQLite(``langgraph-checkpoint-sqlite`` 提供的 AsyncSqliteSaver),
  数据库文件位置由环境变量 ``CHECKPOINT_PATH`` 决定, 默认 ``~/.joblens/checkpoints.db``。
- 同一个 ``thread_id`` 的多次请求共用同一份 state: 第二次调用时, LangGraph 先把上一轮
  的 state 从 checkpoint 恢复出来, 再在它之上做增量更新(多轮记忆 / 断点续跑就是这么来的)。
- 不同 ``thread_id`` 之间彼此不可见, 所以一张编译好的图可以安全地跨请求复用。
- 这一层是**可选**能力: 没装可选依赖、也没有传 thread_id 时, 主流程照走无状态的
  ``joblens.graph.graph``(CLI 与既有测试都在这条路上), 行为分毫不变。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# 进程级单例: (编译好的带 checkpointer 的图, aiosqlite 连接)。
# 连接的寿命跟着进程走, 不主动关闭 —— Web 服务本来就是长驻的。
_cached_graph: Any = None
_cached_conn: Any = None


def get_checkpoint_path() -> Path:
    """checkpoint 那个 SQLite 文件的路径, 可用 CHECKPOINT_PATH 覆盖。"""
    configured = os.getenv("CHECKPOINT_PATH", "").strip()
    if configured:
        db_file = Path(configured).expanduser()
    else:
        db_file = Path.home() / ".joblens" / "checkpoints.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return db_file


def thread_config(thread_id: str) -> dict:
    """LangGraph 的调用约定: thread_id 放在 config["configurable"] 里面。"""
    return {"configurable": {"thread_id": thread_id}}


async def get_persistent_graph() -> Any:
    """取回绑定了 SQLite checkpointer 的编译图(首次调用时懒初始化并缓存)。

    依赖 ``langgraph-checkpoint-sqlite``(pyproject 里已经声明)。在过于精简的环境里
    缺这个包, 就抛出带安装提示的 RuntimeError, 交给 API 层翻译成 503。
    """
    global _cached_graph, _cached_conn
    if _cached_graph is not None:
        return _cached_graph

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "持久化记忆需要可选依赖: pip install langgraph-checkpoint-sqlite"
        ) from exc

    # 拖到这一步才 import build_graph: 万一缺依赖, 连 joblens.memory 本身都还能正常导入。
    from joblens.graph import build_graph

    checkpoint_file = get_checkpoint_path()
    _cached_conn = await aiosqlite.connect(str(checkpoint_file))
    saver = AsyncSqliteSaver(_cached_conn)
    # 老版本必须显式建表, 新版本这一步是幂等的 —— 无脑调一次, 两边都安全。
    if hasattr(saver, "setup"):
        await saver.setup()
    _cached_graph = build_graph(checkpointer=saver)
    return _cached_graph


def reset_persistent_graph() -> None:
    """仅供测试: 丢掉进程内缓存的持久化图与连接。"""
    global _cached_graph, _cached_conn
    _cached_graph = None
    _cached_conn = None


def _jsonable(value: Any) -> Any:
    """把 checkpoint state 里的 Pydantic 模型转成能 JSON 序列化的普通结构。"""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


# 对外透出的 state 字段白名单(内部 channel / 临时对象不直接暴露)。
_PUBLIC_STATE_KEYS = (
    "jd_input",
    "jd_raw_text",
    "jd_structured",
    "gap_analysis",
    "advice",
    "similar_histories",
    "errors",
)


async def get_thread_state(thread_id: str) -> Optional[dict]:
    """读取某 thread_id 最近一次 checkpoint 的 state; 一点记录都没有时返回 None。"""
    compiled = await get_persistent_graph()
    snapshot = await compiled.aget_state(thread_config(thread_id))
    stored_values = getattr(snapshot, "values", None)
    if not stored_values:
        return None
    public_state = {
        key: _jsonable(stored_values[key])
        for key in _PUBLIC_STATE_KEYS
        if key in stored_values
    }
    return {"thread_id": thread_id, "state": public_state}
