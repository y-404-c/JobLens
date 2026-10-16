"""配置读取: API key / 模型名 / base_url 一律从环境变量来, 代码里不写死。

可用的环境变量清单在仓库根目录的 .env.example 里。为了让老配置继续能跑,
当 LLM_API_KEY 缺席时会退一步去读 DEEPSEEK_API_KEY —— 两个里配上一个就行。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 这里先自己拼出项目根目录(pyproject.toml 那一层)再加载 .env, 而不是直接用
# 无参 load_dotenv() —— 后者以"进程启动时的工作目录"为基准, 换个目录启动就会漏读。
# 层级数一下: config.py → joblens → src → 项目根, 所以是 parents[2]。
# (曾经写成 parents[3], 结果读的是上上级目录的 .env —— 文件不存在时 load_dotenv
#  不报错、静默跳过, 所有环境变量都落回默认值, 这种"静默失效"最难排查。)
_ENV_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ENV_ROOT / ".env")


def _env_float(key: str, fallback: float) -> float:
    """读一个浮点型环境变量, 读不到或解析失败就退回 fallback。"""
    try:
        return float(os.getenv(key, str(fallback)).strip())
    except (TypeError, ValueError):
        return fallback


def _env_int(key: str, fallback: int) -> int:
    """读一个整型环境变量, 读不到或解析失败就退回 fallback。"""
    try:
        return int(os.getenv(key, str(fallback)).strip())
    except (TypeError, ValueError):
        return fallback


def get_deepseek_model(temperature: float = 0.3, max_tokens: int = 8192) -> ChatOpenAI:
    """返回一个配置好的 OpenAI-compatible 聊天模型客户端。

    走 OpenAI-compatible 接口(DeepSeek 官方支持它), 而不用专用 SDK。
    trade-off: 专用 SDK 说不定能摸到 DeepSeek 独门特性, 但换成兼容接口之后,
    "将来要换模型(OpenAI / 通义 / 本地 vLLM)"只动 .env 即可, 上层节点的代码一行不改。
    对于一个要长期维护、还得在面试里讲清"可扩展性"的项目, 这点解耦更划算。

    模型名 / base_url / key 三样都取自环境变量:
    - LLM_MODEL     未设置时用 deepseek-chat
    - LLM_BASE_URL  未设置时用 https://api.deepseek.com/v1
    - LLM_API_KEY   未设置时退回 DEEPSEEK_API_KEY
    - LLM_TEMPERATURE / LLM_MAX_TOKENS 设置后会覆盖本函数的默认参数

    temperature=0.3: 抽取/分析这种活要的是稳定、可复现, 不需要它自由发散, 所以压低。
    max_tokens=8192: 这是 DeepSeek 的输出上限。analyze/advise 的结构化结果块头很大
      (十几条带 evidence 的 skill_matches, 再加多条改写建议), 给 4096 会偶发截断,
      于是 with_structured_output 拿到残缺 JSON, 不声不响地返回 None。干脆顶到上限留足余量。
    """
    # 兼容老的变量名 DEEPSEEK_API_KEY
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY (or DEEPSEEK_API_KEY) not set. "
            "Copy .env.example to .env and fill it."
        )

    model_name = os.getenv("LLM_MODEL", "deepseek-chat").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
    # 环境变量只在用户真的配了的时候才覆盖入参, 这样函数签名保持向后兼容
    temperature = _env_float("LLM_TEMPERATURE", temperature)
    max_tokens = _env_int("LLM_MAX_TOKENS", max_tokens)

    # LLM_EXTRA_BODY: 追加到请求体的模型私有参数(JSON 字符串)。
    # 场景: Qwen3.x 这类思考(thinking)模型在思考模式下不支持 tool_choice=required,
    # 而 with_structured_output(function_calling) 恰恰依赖它 —— 配
    # LLM_EXTRA_BODY={"enable_thinking": false} 即可; 用 DeepSeek 时无需配置。
    extra_body = None
    raw_extra = os.getenv("LLM_EXTRA_BODY", "").strip()
    if raw_extra:
        import json as _json

        try:
            extra_body = _json.loads(raw_extra)
        except ValueError as exc:
            raise RuntimeError(f"LLM_EXTRA_BODY 不是合法 JSON: {raw_extra[:80]}") from exc

    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def get_cors_origins() -> list[str]:
    """CORS 放行的前端来源(逗号分隔); 默认只开本机前端, 上生产应显式配白名单。

    一个绕不开的浏览器限制: allow_credentials=True 时, 带凭证的请求不能把 origin 写成 *,
    所以这里的默认值是几个明确的 localhost 来源, 而不是 "*"。
    """
    configured = os.getenv("CORS_ORIGINS", "").strip()
    if not configured:
        return [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    if configured == "*":
        return ["*"]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]
