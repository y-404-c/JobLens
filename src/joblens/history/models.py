"""History 领域的模型定义。"""
from typing import Optional

from pydantic import BaseModel, Field


class HistoryEntry(BaseModel):
    """一条历史分析记录。

    它对应 SQLite 里的 history_entries 表。full_result 由存储层负责跟
    full_result_json(TEXT 列)之间做 dict<->json 的互转; 列表查询时不会加载它
    (留一个空 dict), 只有查详情的查询才会把它填上。
    """
    id: Optional[int] = None
    created_at: str                      # ISO 8601 时间戳
    jd_text: str                         # JD 原文, 留着方便重新分析
    jd_summary: str                      # 拼好的一句话, 列表展示用
    resume_filename: str                 # 只存文件名, 不存 PDF 内容(隐私)
    match_score: Optional[int] = None    # 0-100
    should_apply: Optional[str] = None
    full_result: dict = Field(default_factory=dict)   # 完整分析结果
