"""SQLite 存储层。

直接用 Python 标准库里的 sqlite3, 不引 SQLAlchemy —— 单用户、零配置、嵌入式场景下,
ORM 属于杀鸡用牛刀。
DB 文件默认落在 ~/.joblens/history.db(属于用户数据, 既不进 git, 也不进 Docker 镜像)。
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from joblens.history.models import HistoryEntry

# 建表 DDL: 表名与列名都在持久化契约里, 不能动。
_DDL = """
CREATE TABLE IF NOT EXISTS history_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    jd_text TEXT NOT NULL,
    jd_summary TEXT NOT NULL,
    resume_filename TEXT NOT NULL,
    match_score INTEGER,
    should_apply TEXT,
    full_result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON history_entries(created_at DESC);
"""


def _resolve_db_path() -> Path:
    # 可以用 SQLITE_PATH 覆盖(见 README Configuration), 默认是 ~/.joblens/history.db。
    configured = os.getenv("SQLITE_PATH")
    return Path(configured) if configured else Path.home() / ".joblens" / "history.db"


def _row_to_entry(row: sqlite3.Row, full_result: dict) -> HistoryEntry:
    """把一行查询结果还原成 HistoryEntry(list_entries / get_entry 共用)。"""
    return HistoryEntry(
        id=row["id"],
        created_at=row["created_at"],
        jd_text=row["jd_text"],
        jd_summary=row["jd_summary"],
        resume_filename=row["resume_filename"],
        match_score=row["match_score"],
        should_apply=row["should_apply"],
        full_result=full_result,
    )


class HistoryStorage:
    """历史记录的 SQLite 存储。首次实例化就会自动建表(无需手工 migration)。

    vector_store: 可选的向量库。传了它, create/delete 会顺带写向量/删向量; 不传
    (比如单元测试)就是纯 SQLite, 不会触发 embedding/Chroma(测试因此保持轻量、离线)。
    """

    def __init__(self, db_path: Optional[Path] = None, vector_store=None):
        self.db_path = Path(db_path) if db_path else _resolve_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # 让结果行能按列名取字段
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def create_entry(self, entry: HistoryEntry) -> int:
        """插入一条记录, 返回自增 id。"""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO history_entries
                   (created_at, jd_text, jd_summary, resume_filename,
                    match_score, should_apply, full_result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.created_at,
                    entry.jd_text,
                    entry.jd_summary,
                    entry.resume_filename,
                    entry.match_score,
                    entry.should_apply,
                    json.dumps(entry.full_result, ensure_ascii=False),
                ),
            )
            inserted_id = cursor.lastrowid

        # 顺序有讲究: 先把 SQLite 写好(顺带拿到自增 id), 再写向量库 ——
        # 因为 Chroma 那边的 id 用的就是 SQLite 这个 id。
        if self.vector_store is not None:
            entry.id = inserted_id
            try:
                self.vector_store.index_entry(entry)
            except Exception:
                # 向量化失败不该让"存历史"整件事失败(SQLite 那边已经写成功了)。
                pass
        return inserted_id

    def list_entries(self, limit: int = 50) -> list[HistoryEntry]:
        """列表查询, 按 created_at 倒序。不加载 full_result_json(列表用不到, 省体积)。"""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, jd_text, jd_summary, resume_filename,
                          match_score, should_apply
                   FROM history_entries
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_entry(row, {}) for row in rows]   # 列表不带完整结果

    def get_entry(self, entry_id: int) -> Optional[HistoryEntry]:
        """按 id 取单条完整记录; 不存在就返回 None。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM history_entries WHERE id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row, json.loads(row["full_result_json"]))

    def delete_entry(self, entry_id: int) -> bool:
        """删除一条; 返回是否真删到了(id 不存在则为 False)。同步删向量, 防止"僵尸检索"。"""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM history_entries WHERE id = ?", (entry_id,))
            removed = cursor.rowcount > 0
        if removed and self.vector_store is not None:
            try:
                self.vector_store.remove(entry_id)
            except Exception:
                pass
        return removed


# ---- 进程级单例(API 用); 测试可调 set_storage 注入临时库 ----
_active_storage: Optional[HistoryStorage] = None


def get_storage() -> HistoryStorage:
    """拿到进程内共享的存储实例, 首次调用时创建。"""
    global _active_storage
    if _active_storage is None:
        # 默认存储挂上向量库, 让 create/delete 顺带维护 Chroma。
        from joblens.rag.vector_store import get_vector_store
        _active_storage = HistoryStorage(vector_store=get_vector_store())
    return _active_storage


def set_storage(storage: HistoryStorage) -> None:
    """替换进程级单例(测试用来注入临时库, 或置为 None 复位)。"""
    global _active_storage
    _active_storage = storage


def reindex_missing(storage: HistoryStorage) -> int:
    """平滑升级用: 给"SQLite 里有、向量库里没有"的历史补索引, 返回补了多少条。

    什么时候会缺: M8 之前存下的历史(那会儿还没有向量库), 或者手工删掉
    ~/.joblens/chroma/ 之后重启。
    """
    if storage.vector_store is None:
        return 0
    indexed_ids = storage.vector_store.existing_ids()
    backfilled = 0
    for record in storage.list_entries(limit=10_000):
        if str(record.id) in indexed_ids:
            continue
        complete = storage.get_entry(record.id)   # 得先有 full_result 才能生成 embedding
        if complete is None:
            continue
        storage.vector_store.index_entry(complete)
        backfilled += 1
    return backfilled
