"""结构化 LLM 调用的有限重试封装。

为什么非要有这一层(实测观察):DeepSeek 的 function_calling 时不时不回有效的 tool call,
而 with_structured_output 遇到这种情况不报错, 只是静默地给你一个 None; 调用打得太密时
还会被限流。单发一次本来就不保险, 何况我们端到端串了四个节点, 其中任何一个偶然失手,
整份报告就废了 —— 在 Docker 里反复跑验证过, 每回栽的节点都不一样(这次 analyze、
下次 advise), 可见属于随机性失败, 加一层重试就能兜住。

做法:
- 重试有上限(默认 3 次)。失败判定放得很宽: 拿到 None 算失败, 抛异常也算失败, 都要重试。
- 退避按指数走(base_delay * 2**i, 也就是 0.5s、1.0s、2.0s...), 一来给限流留喘息,
  二来把瞬时抖动摊平。
- 次数耗尽仍然失败就 raise, 由调用方节点记进 state['errors']: 不做无限重试,
  该上报的失败就得浮出来。
"""
import asyncio


async def ainvoke_structured_with_retry(
    structured_model,
    prompt: str,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
):
    """调用 structured_model.ainvoke(prompt); 结果为空或抛异常都算失败, 最多重试有限次。"""
    last_failure = "unknown error"
    for attempt in range(attempts):
        try:
            outcome = await structured_model.ainvoke(prompt)
            if outcome is not None:
                return outcome
            last_failure = "model returned no structured output (possibly truncated)"
        except Exception as exc:  # noqa: BLE001 —— 故意兜底: 各种网络/解析异常都重试
            last_failure = str(exc)
        if attempt + 1 < attempts:
            # 真正的指数退避: 第 attempt 次失败后等待 base_delay * 2**attempt
            await asyncio.sleep(base_delay * (2**attempt))
    raise RuntimeError(f"after {attempts} attempts: {last_failure}")
