"""四个节点的 prompt 模板（全中文）。

贯穿所有节点的 prompt 设计原则:
- 写死 "禁止编造" —— 反幻觉
- 写死 "用 null / 空列表" —— 字段缺失要有确定行为, 不能由模型瞎填
- 只抽硬技能、明确排除软技能 —— 下游的 gap 分析才有区分度
- 输出语言 = 中文 —— 界面是中文, 自然语言字段(strengths/gaps/建议等)必须中文; 而直接
  从 JD 抽取的结构化字段与简历引用保持原文语言(见 _输出语言规则)。
"""

# 三个 prompt 共用的输出语言要求, 拼接在每个模板末尾。
_输出语言规则 = """

重要：用中文（简体）回答。输出中所有自由文本字段必须是中文，包括：
- strengths、critical_gaps、minor_gaps（analyze 阶段）
- rewritten 文本、rationale、should_apply_reason、interview_prep_focus、questions_to_ask_interviewer（advise 阶段）

直接从 JD 抽取的结构化字段（role_title、company、hard_skills、nice_to_have）保持 JD 原文语言——JD 是中文就保留中文，是英文就保留英文，不要翻译。

skill_matches[].evidence 引用简历原文，同样保持原文语言。"""


EXTRACT_JD_PROMPT = """你正在分析一份岗位描述（JD），从中抽取结构化要求。

从下面的 JD 文本中抽取以下内容，务必精确、忠实于原文：

- 禁止编造 JD 中没有提到的技能或要求。
- 某个字段没提到时，用 null（可选字段）或空列表。
- hard_skills：只抽取具体的技术技能（编程语言、框架、工具、协议），不要包含软技能。
- nice_to_have：只收录明确标注为"加分项" / "preferred" / "bonus" / "plus" 的内容。
- key_responsibilities：5-10 条具体的日常工作内容。

JD 文本：
---
{jd_text}
---

现在开始抽取。""" + _输出语言规则


ANALYZE_PROMPT = """你正在把候选人的简历与一份结构化的岗位描述（JD）做对比。

JD（结构化）：
```json
{jd}
```

简历（原始文本）：
---
{resume}
---

你的任务：
1. 对 JD 中的每一条 hard_skill，判断简历体现的程度：matched / partial / missing。
   - matched：简历明确体现了该技能（引用证据原句）。
   - partial：相关但不完全一致（例如 JD 要 React，简历写的是 Vue——记为 partial）。
   - missing：简历中没有任何证据。
2. 找出 3-5 个 strengths：简历超出 JD 要求的亮点。
3. 找出 critical_gaps（最多 5 个）：会实质影响是否投递的关键缺口。
4. 给出 overall_match_score（0-100）：
   - 90+：高度匹配，几乎稳拿面试
   - 70-89：匹配良好，值得投递
   - 50-69：够一够，存在关键缺口
   - <50：明显不匹配

要诚实。不要为了鼓励而虚高分数。""" + _输出语言规则


ADVISE_PROMPT = """你是一位求职顾问，帮助求职者针对特定 JD 打磨简历。

JD（结构化）：
```json
{jd}
```

差距分析：
```json
{gap}
```

简历（原始文本）：
---
{resume}
---

你的任务：
1. 判断 should_apply：
   - strongly_recommend：match_score >= 80，且无 critical_gaps
   - worth_trying：60-79，有缺口但亮点可以弥补
   - stretch：50-59，仅在用户有时间和信心时投递
   - not_recommended：<50，或存在硬伤级缺口

2. 给出 3-5 条 bullet_rewrites：
   - 挑选与这份 JD 最相关的简历 bullet。
   - 逐条改写，突出 JD 中的关键词（仅当底层经历真实存在时）。
   - 重要：禁止编造业绩。只能换一种表述角度，突出已有经历的相关面。
   - 每条改写附一句 rationale（改写理由）。

3. 给出 3-5 条 interview_prep_focus：
   - 结合 JD 要求与缺口，候选人应该重点复习哪些技术主题？
   - 要具体（例如"复习 LangGraph checkpointer 与 thread 持久化机制"，而不是"复习 LangGraph"）。

4. 给出 2-3 条 questions_to_ask_interviewer：
   - 既能体现投入度，又能帮助候选人评估匹配度的问题。
   - 避免"你们公司文化怎么样"这类泛泛的问题。

## 用户过往的相似投递记录

{similar_histories_context}

生成建议时请参考这些过往投递：
- 如果用户以往投相似岗位时总是缺同一类技能（例如总是缺 "Milvus" 或 "gRPC"），点出这个规律："这个缺口在你的多次投递中反复出现，值得花时间系统补齐。"
- 如果某次相似投递的得分明显更高/更低，简要说明这次有什么变化。
- 只有 1 条相似记录时不要硬找规律；只有 2 条及以上相似记录时才评论规律。
- 规律性的观察写进 should_apply_reason 和/或 interview_prep_focus。""" + _输出语言规则
