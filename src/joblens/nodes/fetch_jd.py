"""Node 1: fetch_jd —— 拿到 JD 的纯文本。

几个关键决定:
- 这里不碰 LLM, 纯 IO 加一点启发式, 又快又不花钱。判断"给的是 URL 还是正文"用不上模型。
- 抓取分两层走: 先用 httpx 静态抓(轻、快), 只有失败或正文太短(基本可以断定是 SPA)才退到
  Playwright(开浏览器又重又慢, 属于最后手段)。两层都空手, 才请用户手动粘贴。
- 失败不 raise, 改成把错误写进 state['errors'], 怎么处理交给上层 CLI。节点里 raise 会把
  整张图打断, 用户得不到友好提示。
"""
from joblens.state import AgentState
from joblens.tools import web_scraper

# httpx 抓回的正文若短于该长度, 就认定静态抓取无效(SPA 空壳或被挡), 该上 Playwright 了。
_STATIC_BODY_MIN_CHARS = 500
# 最终正文若短于该长度, 直接算抓取失败。
_FINAL_BODY_MIN_CHARS = 200


async def _scrape_url_layered(target_url: str) -> tuple[str, list[str]]:
    """按层抓 URL: 先用 httpx, 不行或过短再退 Playwright。返回 (文本, 提示列表)。"""
    notes: list[str] = []

    # 第 1 步: 让 httpx 做静态抓取
    static_body = ""
    try:
        static_body = await web_scraper.fetch_and_extract(target_url)
    except Exception as exc:
        notes.append(f"static fetch failed ({exc}), trying Playwright")
    else:
        if len(static_body) >= _STATIC_BODY_MIN_CHARS:
            return static_body, notes
        notes.append(f"static fetch too short ({len(static_body)} chars), trying Playwright")

    # 第 2 步: 交给 Playwright 做动态抓取
    try:
        browser_body = await web_scraper.fetch_with_playwright(target_url)
    except Exception as exc:
        notes.append(f"Playwright fetch failed ({exc})")
        return static_body, notes
    if len(browser_body) >= len(static_body):
        return browser_body, notes
    return static_body, notes  # 动态结果反而更差时, 宁可保留静态那份


async def fetch_jd(state: AgentState) -> dict:
    """取得 JD 纯文本。

    走法:
      1. jd_input 看起来是 URL: 先 httpx, 再 Playwright; 正文依旧不够就报错, 请用户粘原文
      2. jd_input 本身就是正文(>= 200 字符): 直接采信
    """
    raw_input = state["jd_input"]

    if raw_input.startswith(("http://", "https://")):
        body, notes = await _scrape_url_layered(raw_input)
        if len(body) < _FINAL_BODY_MIN_CHARS:
            detail = "; ".join(notes) if notes else "no content"
            return {"errors": [
                f"JD fetch failed ({detail}). This site may be JS-rendered or anti-bot "
                f"protected — please paste the JD text directly."
            ]}
        return {"jd_raw_text": body}

    body = raw_input  # 当作原文处理
    if len(body) < _FINAL_BODY_MIN_CHARS:
        return {"errors": ["JD text too short (<200 chars). Provide the full JD."]}
    return {"jd_raw_text": body}
