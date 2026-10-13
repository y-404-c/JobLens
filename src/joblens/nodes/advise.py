"""Node 4: advise —— 结合 gap 分析与相似历史投递给出建议。

关键决策:
- 三个输入一起给(jd_structured + gap_analysis + resume_raw_text): advise 属于"综合决策"节点。
- RAG: 调 LLM 之前先检索相似历史投递并注入 prompt, 让建议参考用户的历史模式(比如复发缺口)。
  检索失败或结果为空都不该阻断 advise —— 降级成"没有历史上下文"继续跑。
- 检索结果同时写进 state['similar_histories'], 供前端展示。
- 与 extract/analyze 相同: method="function_calling"。
"""
from joblens.config import get_deepseek_model
from joblens.nodes._structured import ainvoke_structured_with_retry
from joblens.prompts.templates import ADVISE_PROMPT
from joblens.rag.retriever import find_similar, format_similar_histories_for_prompt
from joblens.state import AgentState, ResumeAdvice


async def advise(state: AgentState) -> dict:
    """基于 gap 分析 + 相似历史生成简历改写建议 + 面试准备重点。"""
    if not state.get("gap_analysis") or not state.get("jd_structured"):
        return {"errors": ["advise: missing prerequisites (gap_analysis or jd_structured)"]}

    # RAG 检索(失败就降级为空, 不阻断主流程)
    try:
        recalled = find_similar(state["jd_structured"], top_k=3)
    except Exception:
        recalled = []

    llm = get_deepseek_model()
    schema_model = llm.with_structured_output(ResumeAdvice, method="function_calling")

    user_prompt = ADVISE_PROMPT.format(
        jd=state["jd_structured"].model_dump_json(indent=2),
        gap=state["gap_analysis"].model_dump_json(indent=2),
        resume=state.get("resume_raw_text", ""),
        similar_histories_context=format_similar_histories_for_prompt(recalled),
    )
    try:
        parsed = await ainvoke_structured_with_retry(schema_model, user_prompt)
        return {"advice": parsed, "similar_histories": recalled}
    except Exception as exc:
        # 就算 advise 的 LLM 调用失败, 也把检索到的相似历史透出给前端。
        return {"errors": [f"advise failed: {exc}"], "similar_histories": recalled}
