"""职探 JobLens 的主图: fetch_jd → extract → analyze → advise。

模块级的 `graph` 不带 checkpointer —— 单次分析跑完即止, 行为跟 MVP 一致。
要做多轮 / 持久化记忆时, 用 `build_graph(checkpointer=...)` 编译一张带 checkpointer
的图, 调用时传 config={"configurable": {"thread_id": ...}} —— 同一个 thread_id
的历史 state 会从 checkpoint 存储里恢复(细节见 joblens.memory)。
"""
from typing import Optional

from langgraph.graph import StateGraph, START, END

from joblens.state import AgentState
from joblens.nodes.fetch_jd import fetch_jd
from joblens.nodes.extract import extract
from joblens.nodes.analyze import analyze
from joblens.nodes.advise import advise


def build_graph(checkpointer: Optional[object] = None):
    """构建职探 JobLens 主图: fetch_jd → extract → analyze → advise(四个节点一线串起)。

    Args:
        checkpointer: 可选的 LangGraph checkpointer (MemorySaver /
            SqliteSaver 等)。传了它, 图就按 ``thread_id`` 把 state 持久化,
            从而支持断点续跑与多轮记忆; 不传就是无状态图(向后兼容,
            CLI 与现有测试走的都是这条路径)。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_jd", fetch_jd)
    workflow.add_node("extract", extract)
    workflow.add_node("analyze", analyze)
    workflow.add_node("advise", advise)

    workflow.add_edge(START, "fetch_jd")
    workflow.add_edge("fetch_jd", "extract")
    workflow.add_edge("extract", "analyze")
    workflow.add_edge("analyze", "advise")
    workflow.add_edge("advise", END)

    return (
        workflow.compile()
        if checkpointer is None
        else workflow.compile(checkpointer=checkpointer)
    )


# 全局无状态实例, CLI 和 web server 的默认路径都复用它。
# 需要持久化记忆时不要直接把它换掉(那会变成跨请求共用一个会话),
# 而是去 joblens.memory.get_persistent_graph() 拿带 checkpointer 的图。
graph = build_graph()
