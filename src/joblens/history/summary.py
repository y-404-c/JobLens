"""生成 JD 摘要 —— 拼出一句话, 供历史列表展示。"""
from joblens.state import JDRequirements


def generate_summary(requirements: JDRequirements) -> str:
    """把结构化 JD 拼成一句话摘要。

    有公司名: "公司 · 岗位 · 地点"; 没有公司名: 兜底成 "岗位 · 薪资"。
    role_title 是必填字段(垃圾输入会被前置的 200 字符保护拦下), 所以这里不再兜底。
    """
    if requirements.company:
        return f"{requirements.company} · {requirements.role_title} · {requirements.location or '地点未知'}"
    return f"{requirements.role_title} · {requirements.salary_range or '薪资未知'}"
