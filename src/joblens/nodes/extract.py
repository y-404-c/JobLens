"""Node 2: extract —— 从 JD 原文抽取结构化需求(JDRequirements)。

关键决策:
- 走 .with_structured_output(JDRequirements): 让 LangChain 把 Pydantic schema 翻成
  模型的 function-calling / json schema, 模型输出会被强制校验成 JDRequirements 实例。
  不合格会自动 retry —— 这是"反幻觉 + 数据可预期"的核心机制。
- 出错只往 state['errors'] 追加, 不 raise: 理由同 fetch_jd, 让整张图跑完、
  由上层决定如何呈现, 而不是中途炸掉。
"""
from joblens.config import get_deepseek_model
from joblens.nodes._structured import ainvoke_structured_with_retry
from joblens.prompts.templates import EXTRACT_JD_PROMPT
from joblens.state import AgentState, JDRequirements


async def extract(state: AgentState) -> dict:
    """从 JD 原文抽取结构化需求。"""
    if not state.get("jd_raw_text"):
        return {"errors": ["extract: no jd_raw_text to process"]}

    llm = get_deepseek_model()
    # method="function_calling": langchain-openai 1.x 默认用 method="json_schema"
    # (OpenAI 的 structured outputs, 走 response_format), 但 DeepSeek 的 OpenAI-compatible
    # 端点不支持它, 会返回 400 "This response_format type is unavailable now"。
    # DeepSeek 支持的是 function/tool calling, 所以显式改成 function_calling, 让 schema
    # 通过 tool 参数下发。(对应蓝图第 15 节的已知坑。)
    schema_model = llm.with_structured_output(JDRequirements, method="function_calling")

    user_prompt = EXTRACT_JD_PROMPT.format(jd_text=state["jd_raw_text"])
    try:
        parsed = await ainvoke_structured_with_retry(schema_model, user_prompt)
        return {"jd_structured": parsed}
    except Exception as exc:
        return {"errors": [f"extract failed: {exc}"]}
