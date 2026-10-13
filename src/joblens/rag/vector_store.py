"""Chroma 向量库封装。

位置: ~/.joblens/chroma/(和 SQLite 同目录, 一起走 volume 持久化)。
Collection: "jd_history", cosine 距离空间。

底层方法(add/query/delete/count)只收/发 embedding 向量, 不碰 embedder ——
方便测试传假向量, 也让"存什么向量"的策略集中在 index_entry 一处。
"""
import os
from pathlib import Path
from typing import Optional

from joblens.history.models import HistoryEntry
from joblens.rag.embedder import embed_text, build_embedding_text_from_dict

_COLLECTION = "jd_history"


def _resolve_chroma_dir() -> Path:
    # 可用 CHROMA_DIR 覆盖(见 README Configuration), 默认 ~/.joblens/chroma。
    configured = os.getenv("CHROMA_DIR")
    return Path(configured) if configured else Path.home() / ".joblens" / "chroma"


class VectorStore:
    """Chroma 持久化 collection 的薄封装: 底层收发向量, 上层从 HistoryEntry 入库。"""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else _resolve_chroma_dir()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection = None

    def _handle(self):
        """惰性建客户端并取到 collection。"""
        if self._collection is None:
            import chromadb
            chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = chroma_client.get_or_create_collection(
                name=_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    # ---- 低层接口: 只负责收发向量(测试可以塞假向量) ----
    def add(self, id: str, document: str, metadata: dict, embedding: list[float]) -> None:
        """写入(或覆盖)一条记录。"""
        self._handle().upsert(
            ids=[str(id)], documents=[document], metadatas=[metadata], embeddings=[embedding]
        )

    def query(self, embedding: list[float], top_k: int) -> dict:
        """按向量查最近邻, 返回 Chroma 的原始结果 dict。"""
        return self._handle().query(query_embeddings=[embedding], n_results=top_k)

    def delete(self, id: str) -> None:
        """按 id 删除一条记录。"""
        self._handle().delete(ids=[str(id)])

    def count(self) -> int:
        """当前 collection 的记录数。"""
        return self._handle().count()

    def existing_ids(self) -> set[str]:
        """取出 collection 内全部 id。"""
        return set(self._handle().get()["ids"])

    # ---- 高层接口: 从 HistoryEntry 生成向量并写库 ----
    def index_entry(self, entry: HistoryEntry) -> None:
        """给一条历史记录生成 embedding 并写入(需要 full_result['jd'])。"""
        jd_payload = (entry.full_result or {}).get("jd")
        if not jd_payload or entry.id is None:
            return
        embed_source = build_embedding_text_from_dict(jd_payload)
        self.add(
            id=str(entry.id),
            document=embed_source,
            metadata={
                "history_id": entry.id,
                "jd_summary": entry.jd_summary,
                "match_score": entry.match_score if entry.match_score is not None else -1,
                "should_apply": entry.should_apply or "",
                "created_at": entry.created_at,
            },
            embedding=embed_text(embed_source),
        )

    def remove(self, history_id: int) -> None:
        """按 history_id 删除对应向量。"""
        self.delete(str(history_id))


# ---- 进程级单例(测试可整体替换) ----
_singleton: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """取进程级单例(首次调用时创建)。"""
    global _singleton
    if _singleton is None:
        _singleton = VectorStore()
    return _singleton


def set_vector_store(store: Optional[VectorStore]) -> None:
    """替换/清空进程级单例。"""
    global _singleton
    _singleton = store
