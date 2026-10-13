"""LangGraph checkpointer 持久化记忆的单元测试。

不依赖 langgraph-checkpoint-sqlite: 用 langgraph 自带的内存 saver 验证
"同一 thread_id 跨调用恢复 state / 不同 thread_id 互相隔离" 这一核心契约;
SQLite 落盘只是同一接口的另一个后端。
"""
import pytest

from joblens import graph as graph_module
from joblens.graph import build_graph
from joblens.memory import thread_config, _jsonable
from joblens.state import JDRequirements


def test_thread_config_shape():
    cfg = thread_config("sess-42")
    assert cfg == {"configurable": {"thread_id": "sess-42"}}


def test_checkpoint_path_env_override(monkeypatch, tmp_path):
    from joblens import memory

    target = tmp_path / "cp" / "checkpoints.db"
    monkeypatch.setenv("CHECKPOINT_PATH", str(target))
    assert memory.get_checkpoint_path() == target
    assert target.parent.is_dir()  # 父目录应自动建好


def test_checkpoint_path_default(monkeypatch):
    from joblens import memory
    from pathlib import Path

    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    assert memory.get_checkpoint_path() == Path.home() / ".joblens" / "checkpoints.db"


def test_jsonable_dumps_pydantic():
    req = JDRequirements(
        role_title="Agent 工程师", seniority="intern",
        hard_skills=["LangGraph"], key_responsibilities=["r"],
    )
    out = _jsonable({"jd_structured": req, "nested": [req], "plain": 1})
    assert out["jd_structured"]["role_title"] == "Agent 工程师"
    assert out["nested"][0]["seniority"] == "intern"
    assert out["plain"] == 1


@pytest.mark.asyncio
async def test_checkpointer_same_thread_accumulates(monkeypatch):
    """同一 thread_id 跑两轮: errors 经 reducer 从 4 追加到 8, 证明历史 state 被恢复。"""
    # InMemorySaver 是 1.x 的名字, 老版本叫 MemorySaver。
    try:
        from langgraph.checkpoint.memory import InMemorySaver as Saver
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver as Saver

    labels = ["fetch", "extract", "analyze", "advise"]
    for name, label in zip(("fetch_jd", "extract", "analyze", "advise"), labels):
        def _make(_label):
            def _node(state):
                return {"errors": [_label]}
            return _node
        monkeypatch.setattr(graph_module, name, _make(label))

    g = build_graph(checkpointer=Saver())
    cfg = thread_config("thread-A")

    await g.ainvoke({"jd_input": "q1", "resume_raw_text": "cv"}, cfg)
    snap1 = await g.aget_state(cfg)
    assert snap1.values["errors"] == labels

    # 同 thread 第二轮: 不是覆盖, 而是在 checkpoint 恢复的旧 state 上追加。
    await g.ainvoke({"jd_input": "q2", "resume_raw_text": "cv"}, cfg)
    snap2 = await g.aget_state(cfg)
    assert snap2.values["errors"] == labels + labels


@pytest.mark.asyncio
async def test_checkpointer_different_threads_isolated(monkeypatch):
    try:
        from langgraph.checkpoint.memory import InMemorySaver as Saver
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver as Saver

    for name in ("fetch_jd", "extract", "analyze", "advise"):
        monkeypatch.setattr(graph_module, name, lambda state: {"errors": ["x"]})

    g = build_graph(checkpointer=Saver())
    await g.ainvoke({"jd_input": "q", "resume_raw_text": "cv"}, thread_config("thread-A"))

    # 新 thread_id 看不到 A 的历史, 只有自己这一轮的 4 条。
    snap_b = await g.aget_state(thread_config("thread-B"))
    assert not snap_b.values
    await g.ainvoke({"jd_input": "q", "resume_raw_text": "cv"}, thread_config("thread-B"))
    snap_b = await g.aget_state(thread_config("thread-B"))
    assert snap_b.values["errors"] == ["x", "x", "x", "x"]
