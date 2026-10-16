"""文本向量化层: 两种后端, 由环境变量决定, 对上层接口不变。

- API 模式(配置了 ``EMBEDDING_BASE_URL``): 走 OpenAI 兼容的 /embeddings 接口
  (DashScope / 自建 vLLM 都支持), 模型名用 ``EMBEDDING_MODEL``。
  好处: 不用本地下载模型、不吃本机算力、换模型只动 .env。
- 本地模式(未配 ``EMBEDDING_BASE_URL``): sentence-transformers 加载本地模型,
  默认 ``paraphrase-multilingual-MiniLM-L12-v2``, 离线可用、零调用成本。

关键约束: 检索端(find_similar)与入库端(index_entry)必须落在同一个语义空间 ——
两种模式都把向量做 L2 归一化, 配合 Chroma 的 cosine 距离空间使用。
**切换 embedding 后端后必须清空 CHROMA_DIR 重建**, 否则新旧向量维度/空间不一致。
"""
import math
import os
from typing import Optional

# 确保 .env 已加载(load_dotenv 的副作用在 config 模块里)。
# 若有人绕过 config 直接 import 本模块, 不加这行就会读不到 EMBEDDING_BASE_URL,
# 静默回落到本地模式去连 HuggingFace —— 实测踩过, 显式声明依赖。
import joblens.config  # noqa: F401

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 进程级缓存: API 客户端与本地模型都只初始化一次
_api_client = None
_loaded_model = None


def _normalize(vector: list[float]) -> list[float]:
    """L2 归一化: 让"余弦相似度"退化成点积, 与本地模式行为对齐。"""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def _get_api_client():
    global _api_client
    if _api_client is None:
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_BASE_URL 已配置, 但缺少 EMBEDDING_API_KEY / LLM_API_KEY"
            )
        
        _api_client = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip(),
            base_url=os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/"),
            api_key=api_key,
            # 关掉 tiktoken 预分词: 它首用要从 OpenAI CDN 下载词表(国内网络易超时),
            # 且 OpenAI 的 tokenizer 对 Qwen embedding 模型本来就不准 —— 直接发原文即可。
            check_embedding_ctx_length=False,
        )
    return _api_client


def _embed_api(text: str) -> list[float]:
    """API 模式: 调 OpenAI 兼容 /embeddings 接口取回向量。"""
    vector = _get_api_client().embed_documents([text])[0]
    return _normalize([float(x) for x in vector])


def _load_local_model():
    global _loaded_model
    if _loaded_model is None:
        from sentence_transformers import SentenceTransformer

        resolved_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        _loaded_model = SentenceTransformer(resolved_name)
    return _loaded_model


def _embed_local(text: str) -> list[float]:
    """本地模式: sentence-transformers 编码, normalize_embeddings 直接出归一化向量。"""
    vector = _load_local_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_text(text: str) -> list[float]:
    """把文本编码成归一化向量(供 cosine 检索使用)。后端由 EMBEDDING_BASE_URL 决定。"""
    if os.getenv("EMBEDDING_BASE_URL", "").strip():
        return _embed_api(text)
    return _embed_local(text)


def build_embedding_text(
    role_title: str,
    hard_skills: list[str],
    domain: Optional[str],
    seniority: str,
) -> str:
    """生成用于 embedding 的结构化文本。

    用结构化字段而不是 JD 原文: 降噪(去掉薪资/公司套话)、维度可控、可解释。
    检索端和入库端必须用同一套格式, 否则两边落不到同一个语义空间。
    """
    return "\n".join([
        role_title,
        "Skills required: " + ", ".join(hard_skills),
        "Domain: " + (domain or "general"),
        "Seniority: " + seniority,
    ])


def build_embedding_text_from_dict(jd: dict) -> str:
    """从 full_result['jd'] 这类 dict 生成 embedding 文本。"""
    return build_embedding_text(
        role_title=jd.get("role_title", ""),
        hard_skills=jd.get("hard_skills", []) or [],
        domain=jd.get("domain"),
        seniority=jd.get("seniority", "unspecified"),
    )
