"""检索 top-k 相似历史投递。"""
from typing import Callable, Optional

from joblens.state import JDRequirements, SimilarHistory
from joblens.rag.embedder import embed_text, build_embedding_text
from joblens.rag.vector_store import VectorStore, get_vector_store

# 相似度没过这条线的结果一律丢掉, 弱相关的别带进上下文(见 notes)。
_SIMILARITY_CUTOFF = 0.65


def find_similar(
    jd_structured: JDRequirements,
    top_k: int = 3,
    exclude_id: Optional[int] = None,
    threshold: float = _SIMILARITY_CUTOFF,
    store: Optional[VectorStore] = None,
    embedder: Callable[[str], list[float]] = embed_text,
) -> list[SimilarHistory]:
    """检索与当前 JD 最相似的历史投递。

    - query 向量用与入库端相同的结构化文本格式生成。
    - 只返回相似度 > threshold 的(弱相关的不进 prompt)。
    - exclude_id: 若当前这条已在库里, 用它排除自己。
    """
    store = store or get_vector_store()
    if store.count() == 0:
        return []

    query_text = build_embedding_text(
        role_title=jd_structured.role_title,
        hard_skills=jd_structured.hard_skills,
        domain=jd_structured.domain,
        seniority=jd_structured.seniority,
    )
    # 多取一个, 给"排除自己"留余量
    fetch_k = top_k + 1 if exclude_id is not None else top_k
    raw = store.query(embedder(query_text), top_k=fetch_k)

    id_list = raw.get("ids", [[]])[0]
    distance_list = raw.get("distances", [[]])[0]
    metadata_list = raw.get("metadatas", [[]])[0]

    hits: list[SimilarHistory] = []
    for record_id, distance, payload in zip(id_list, distance_list, metadata_list):
        history_id = int(record_id)
        if exclude_id is not None and history_id == exclude_id:
            continue
        cosine_sim = 1.0 - float(distance)   # cosine 距离 -> 相似度
        if cosine_sim < threshold:
            continue
        raw_score = payload.get("match_score")
        hits.append(
            SimilarHistory(
                history_id=history_id,
                jd_summary=payload.get("jd_summary", ""),
                match_score=None if raw_score in (None, -1) else int(raw_score),
                should_apply=payload.get("should_apply") or None,
                similarity_score=round(cosine_sim, 3),
                created_at=payload.get("created_at", ""),
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def format_similar_histories_for_prompt(similar: list[SimilarHistory]) -> str:
    """把检索结果拼成注入 advise prompt 的文本。"""
    if not similar:
        return "(This is the user's first application in this domain; no historical context available.)"
    rows = []
    for position, item in enumerate(similar, 1):
        score = item.match_score if item.match_score is not None else "?"
        rows.append(
            f"{position}. {item.jd_summary} — match_score={score}, should_apply={item.should_apply or '?'}, "
            f"similarity={int(item.similarity_score * 100)}%, applied {item.created_at}"
        )
    return "\n".join(rows)
