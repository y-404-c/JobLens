"""CLI 入口: python -m joblens.cli --jd <URL_or_TEXT> --resume <PDF_path>

几点设计说明:
- 参数交给 argparse; --jd / --resume 必填, --output / --output-md 可选。
- graph.ainvoke() 是 async 的, 而 main() 是同步的, 所以用 asyncio.run() 桥一下
  (整个进程只调一次 asyncio.run, 复习 01-notes 概念⑤)。
- 除了给机器读的 JSON, 也可以产出给人读的 Markdown 报告(--output-md)。
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from joblens.graph import graph
from joblens.tools.pdf_parser import extract_pdf_text


async def run(jd_input: str, resume_pdf: Path) -> dict:
    """跑完整张图, 返回组装好的报告 dict。"""
    resume_text = extract_pdf_text(resume_pdf)
    seed_state = {
        "jd_input": jd_input,
        "resume_raw_text": resume_text,
    }
    result_state = await graph.ainvoke(seed_state)

    jd_obj = result_state.get("jd_structured")
    gap_obj = result_state.get("gap_analysis")
    advice_obj = result_state.get("advice")

    return {
        "jd": jd_obj.model_dump() if jd_obj else None,
        "gap": gap_obj.model_dump() if gap_obj else None,
        "advice": advice_obj.model_dump() if advice_obj else None,
        "errors": result_state.get("errors", []),
    }


def render_markdown(report: dict) -> str:
    """把报告 dict 渲染成人类可读的 Markdown。"""
    jd_data = report.get("jd")
    gap_data = report.get("gap")
    advice_data = report.get("advice")
    out: list[str] = []

    if jd_data:
        role = jd_data.get("role_title", "?")
        employer = jd_data.get("company") or "未知公司"
        out.extend([f"# 职探 JobLens 报告 — {role} @ {employer}", ""])
    else:
        out.extend(["# 职探 JobLens 报告", ""])

    if report.get("errors"):
        out.extend(["## ⚠️ 运行错误", ""])
        out.extend(f"- {err}" for err in report["errors"])
        out.extend([""])

    if jd_data:
        out.extend(["## 岗位需求", ""])
        if jd_data.get("location"):
            out.append(f"- **地点**: {jd_data['location']}")
        if jd_data.get("seniority"):
            out.append(f"- **资历**: {jd_data['seniority']}")
        if jd_data.get("salary_range"):
            out.append(f"- **薪资**: {jd_data['salary_range']}")
        if jd_data.get("hard_skills"):
            out.append(f"- **硬技能**: {', '.join(jd_data['hard_skills'])}")
        if jd_data.get("nice_to_have"):
            out.append(f"- **加分项**: {', '.join(jd_data['nice_to_have'])}")
        out.append("")

    if gap_data:
        out.extend(
            ["## 匹配度分析", "", f"**总体匹配度: {gap_data.get('overall_match_score')}/100**", ""]
        )
        if gap_data.get("skill_matches"):
            out.extend(["| 技能 | 状态 | 证据 |", "| --- | --- | --- |"])
            for match in gap_data["skill_matches"]:
                evidence = (match.get("evidence") or "").replace("|", "\\|").replace("\n", " ")
                out.append(f"| {match['skill']} | {match['status']} | {evidence} |")
            out.append("")
        if gap_data.get("strengths"):
            out.extend(["### 强项", ""])
            out.extend(f"- {strength}" for strength in gap_data["strengths"])
            out.extend([""])
        if gap_data.get("critical_gaps"):
            out.extend(["### 关键缺口", ""])
            out.extend(f"- {gap_text}" for gap_text in gap_data["critical_gaps"])
            out.extend([""])

    if advice_data:
        out.extend(["## 建议", "", f"**是否投递: {advice_data.get('should_apply')}**", ""])
        if advice_data.get("should_apply_reason"):
            out.extend([advice_data["should_apply_reason"], ""])
        if advice_data.get("bullet_rewrites"):
            out.extend(["### 简历改写建议", ""])
            for idx, rewrite in enumerate(advice_data["bullet_rewrites"], 1):
                out.extend(
                    [
                        f"**{idx}. 原文**: {rewrite['original']}",
                        f"   - **改写**: {rewrite['rewritten']}",
                        f"   - **理由**: {rewrite['rationale']}",
                        "",
                    ]
                )
        if advice_data.get("interview_prep_focus"):
            out.extend(["### 面试准备重点", ""])
            out.extend(f"- {focus}" for focus in advice_data["interview_prep_focus"])
            out.extend([""])
        if advice_data.get("questions_to_ask_interviewer"):
            out.extend(["### 反问面试官", ""])
            out.extend(f"- {question}" for question in advice_data["questions_to_ask_interviewer"])
            out.extend([""])

    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    """搭出 CLI 的参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="joblens",
        description="职探 JobLens: 把一份岗位 JD 和你的简历做深度对比, 输出匹配度、改写建议和面试重点。",
        epilog=(
            "示例:\n"
            "  # JD 直接给原文, 报告打到终端\n"
            '  uv run python -m joblens.cli --jd "岗位职责: ..." --resume examples/sample_resume.pdf\n\n'
            "  # JD 给 URL, 同时存下 JSON 与 Markdown\n"
            '  uv run python -m joblens.cli --jd "https://jobs.example.com/123" \\\n'
            "      --resume ./my_resume.pdf --output report.json --output-md report.md\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jd", required=True, help="JD 的 URL 或直接粘贴的原文")
    parser.add_argument("--resume", required=True, type=Path, help="简历 PDF 路径")
    parser.add_argument("--output", type=Path, help="把 JSON 报告存到此路径")
    parser.add_argument("--output-md", type=Path, help="把 Markdown 报告存到此路径")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.resume.exists():
        parser.error(f"简历 PDF 不存在: {args.resume}")

    analysis = asyncio.run(run(args.jd, args.resume))

    payload = json.dumps(analysis, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"JSON 报告已保存到 {args.output}")
    if args.output_md:
        args.output_md.write_text(render_markdown(analysis), encoding="utf-8")
        print(f"Markdown 报告已保存到 {args.output_md}")
    if not args.output and not args.output_md:
        print(payload)

    # 只要带了错误就以非零退出码收场, 方便脚本/CI 判定成败
    return 1 if analysis.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
