"""投递历史存档包: SQLite 存储每一次分析的完整结果。

- models.py   HistoryEntry 数据类(与表结构一一对应)
- storage.py  HistoryStorage: 建表 / 增删查, 可选联动向量库同步索引
- summary.py  把结构化 JD 拼成一句话摘要(列表页展示用)
"""
