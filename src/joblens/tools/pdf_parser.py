"""PDF 简历解析工具。

选型 trade-off:
- 主用 pdfplumber, 不用 pypdf: pypdf 碰多列排版会按物理行把两栏文字混在一起(左栏一句 +
  右栏一句拼成一行), 语义全乱; pdfplumber 按坐标聚类, 保序更可靠。简历偶尔是双栏, 值这个依赖。
- fallback 用 pypdfium2, 不用 LandingAI 之类付费 API: MVP 不引入外部收费依赖。
  pypdfium2 本来就在 pdfplumber 的依赖链里(白得), 底层是 Google 的 PDFium 引擎。

为什么需要 fallback(真实踩到的坑):
  有些 PDF(比如 Markdown 导出的简历)靠"重复描边"来假装粗体, pdfplumber 会把每个粗体字形
  抽两遍 —— "Technical Skills" 变成 "TTeecchhnniiccaall SSkkiillllss"。
  这种双写会污染喂给 LLM 的简历文本。检测办法: 统计"连续重复字符"所占比例, 正常英文文本
  约 3-4%, 双写会飙到 13%+。超过阈值就换 pypdfium2 重抽(实测同一文件它只有 1.8%, 干净)。

MVP 限制:
- 只支持文字版 PDF, 扫描版不支持(要 OCR, Phase 2 再加)。
- 假定单列排版, 双列/图片版可能失败(降级方案: 让用户改成 docx -> 单列 pdf)。
"""
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

# 连续重复字符比例高于这个值, 判定为"粗体双写"污染, 触发 fallback。
# 正常英文 ~3-4%, 中文更低; 双写案例实测 13.5%, 8% 能把两者干净分开。
_DUP_RATIO_CUTOFF = 0.08


def _doubled_char_ratio(text: str) -> float:
    """非空白字符中与前一字符相同的比例, 用来嗅探"粗体双写"污染。"""
    visible = [ch for ch in text if not ch.isspace()]
    if len(visible) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(visible)) if visible[i] == visible[i - 1])
    return repeats / len(visible)


def _read_pages_pdfplumber(pdf_path: Path) -> str:
    """用 pdfplumber 逐页取文字, 页与页之间以空行分隔。"""
    page_texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            page_texts.append(extracted.strip())
    return "\n\n".join(chunk for chunk in page_texts if chunk)


def _read_pages_pdfium(pdf_path: Path) -> str:
    """用 pypdfium2 逐页取文字, 页与页之间以空行分隔。"""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        raw_pages = [page.get_textpage().get_text_range() for page in pdf]
    finally:
        pdf.close()
    return "\n\n".join(chunk.strip() for chunk in raw_pages if chunk.strip())


def extract_pdf_text(pdf_path: Path) -> str:
    """提取 PDF 全文, 页与页之间用空行分隔。

    默认走 pdfplumber; 一旦测出粗体双写污染, 自动 fallback 到 pypdfium2。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf file, got {pdf_path.suffix}")

    primary_text = _read_pages_pdfplumber(pdf_path)

    if _doubled_char_ratio(primary_text) > _DUP_RATIO_CUTOFF:
        # pdfplumber 这份带着双写污染, 换 pypdfium2 重抽一遍
        fallback_text = _read_pages_pdfium(pdf_path)
        # 只有 fallback 确实更干净才采用(个别文件 pypdfium2 反而更差)
        if _doubled_char_ratio(fallback_text) < _doubled_char_ratio(primary_text):
            return fallback_text

    return primary_text
