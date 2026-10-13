"""Node 3: analyze —— 拿 JD 需求对比简历原文, 输出 gap 分析。

关键决策:
- 输入用 state['jd_structured'](Node 2 的结构化产出)而不是 jd_raw_text: 结构化之后字段
  清晰, 模型对比时不必再从原文里"找需求", 更聚焦、也更准。
- 简历这边用 raw_text(不结构化): 简历格式五花八门, 硬做结构化反而丢信息; 让模型直接读
  全文去找证据(evidence)更灵活。
- 与 extract 相同: method="function_calling"(DeepSeek 不支持默认的 json_schema)。
"""
from joblens.config import get_deepseek_model
from joblens.nodes._structured import ainvoke_structured_with_retry
from joblens.prompts.templates import ANALYZE_PROMPT
from joblens.state import AgentState, GapAnalysis


async def analyze(state: AgentState) -> dict:
    """对比 JD 需求与简历原文, 输出 gap 分析。"""
    if not state.get("jd_structured") or not state.get("resume_raw_text"):
        return {"errors": ["analyze: missing prerequisites (jd_structured or resume_raw_text)"]}

    llm = get_deepseek_model()
    schema_model = llm.with_structured_output(GapAnalysis, method="function_calling")

    # 用 model_dump_json 把 JDRequirements 序列化成规整 JSON 再塞进 prompt:
    # 字段名一目了然, 模型读起来更稳(见 02-notes 决策)。
    user_prompt = ANALYZE_PROMPT.format(
        jd=state["jd_structured"].model_dump_json(indent=2),
        resume=state["resume_raw_text"],
    )
    try:
        parsed = await ainvoke_structured_with_retry(schema_model, user_prompt)
        return {"gap_analysis": parsed}
    except Exception as exc:
        return {"errors": [f"analyze failed: {exc}"]}
