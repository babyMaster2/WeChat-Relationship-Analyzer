from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = re.compile(
    r"喜欢|爱你|想你|亲亲|抱抱|宝贝|宝宝|晚安|早安|在一起|分手|生气|对不起|抱歉|"
    r"误会|解释|和好|见面|陪你|担心|到家|吃饭|早点睡|不理你|冷淡|敷衍|未来|关系"
)


def validate_local_config(config: dict[str, Any]) -> None:
    if config.get("analysis_backend", "local") != "local":
        raise RuntimeError("隐私保护：关系分析仅允许 analysis_backend=local。")


def _body(message: dict[str, Any]) -> str:
    value = message.get("transcript") if message.get("type") == "voice" else message.get("text")
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.startswith("<") and ("<msg" in text[:120] or "<?xml" in text[:120]):
        return ""
    return text


def _side(message: dict[str, Any]) -> str:
    if message.get("direction") == "sent" or message.get("sender") == "我":
        return "我"
    if message.get("direction") == "received" or message.get("sender") == "对方":
        return "对方"
    return "未知"


def _line(message: dict[str, Any], limit: int = 140) -> str:
    text = _body(message)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{message.get('time', '未知时间')}｜{_side(message)}：{text}"


def build_evidence_pack(timeline: dict[str, Any], max_chars: int = 60_000) -> str:
    messages = sorted(
        [item for item in timeline.get("messages", []) if isinstance(item, dict)],
        key=lambda item: (int(item.get("timestamp", 0)), str(item.get("id", ""))),
    )
    readable = [message for message in messages if _body(message)]
    if not messages:
        return "# 证据包\n\n没有可分析的消息。"

    monthly: dict[str, Counter[str]] = defaultdict(Counter)
    daily: Counter[str] = Counter()
    session_starts: Counter[str] = Counter()
    response: dict[str, list[int]] = defaultdict(list)
    for message in messages:
        who = _side(message)
        month = str(message.get("time", "未知"))[:7]
        monthly[month][f"{who}_消息"] += 1
        monthly[month][f"{who}_字数"] += len(_body(message))
        monthly[month][f"{who}_语音"] += int(message.get("type") == "voice")
        daily[str(message.get("time", "未知"))[:10]] += 1
    if messages:
        session_starts[_side(messages[0])] += 1
    for previous, current in zip(messages, messages[1:]):
        gap = int(current.get("timestamp", 0)) - int(previous.get("timestamp", 0))
        if gap >= 6 * 3600:
            session_starts[_side(current)] += 1
        if _side(previous) != _side(current) and 0 <= gap <= 6 * 3600:
            response[_side(current)].append(gap)

    lines = [
        "# 本地关系分析证据包",
        "",
        f"- 联系人：{timeline.get('session', '未知')}",
        f"- 时间：{messages[0].get('time')} 至 {messages[-1].get('time')}",
        f"- 消息总数：{len(messages)}；可读文字或语音转写：{len(readable)}",
        f"- 以静默 6 小时划分的会话发起：我 {session_starts['我']} 次，对方 {session_starts['对方']} 次",
        "- 发送者已经由导出元数据和媒体 XML 校正为“我/对方”；不得使用被污染的旧昵称。",
        "",
        "## 月度指标",
        "",
        "| 月份 | 我消息 | 对方消息 | 我字数 | 对方字数 | 我语音 | 对方语音 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(monthly):
        c = monthly[month]
        lines.append(
            f"| {month} | {c['我_消息']} | {c['对方_消息']} | {c['我_字数']} | {c['对方_字数']} | "
            f"{c['我_语音']} | {c['对方_语音']} |"
        )
    lines.extend(["", "## 响应节奏"])
    for who in ("我", "对方"):
        values = response[who]
        value = f"{median(values) / 60:.1f} 分钟" if values else "无法计算"
        lines.append(f"- {who}跨方响应中位数：{value}（{len(values)} 次）")
    lines.extend(["", "## 活跃日期"])
    lines.extend(f"- {day}：{count} 条" for day, count in daily.most_common(15))

    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in readable:
        by_month[str(message.get("time", "未知"))[:7]].append(message)
    lines.extend(["", "## 每月首尾证据"])
    for month in sorted(by_month):
        rows = by_month[month]
        lines.append(f"### {month}")
        for message in rows[:4] + rows[-4:]:
            lines.append(f"- {_line(message)}")

    lines.extend(["", "## 关系关键词及上下文"])
    hits = [index for index, message in enumerate(messages) if KEYWORDS.search(_body(message))]
    quota: Counter[tuple[str, str]] = Counter()
    selected: list[int] = []
    for index in hits:
        message = messages[index]
        bucket = (str(message.get("time", "未知"))[:7], _side(message))
        if quota[bucket] < 6:
            quota[bucket] += 1
            selected.append(index)
    for index in selected:
        lines.append(f"- 命中：{_line(messages[index])}")
        for nearby in messages[max(0, index - 1) : min(len(messages), index + 2)]:
            if nearby is not messages[index] and _body(nearby):
                lines.append(f"  - 上下文：{_line(nearby, 100)}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n> 证据包已按本机上下文上限截断；结论必须降低置信度。"
    return text


def build_prompt(session_name: str, evidence: str) -> str:
    return f"""你是一名谨慎的私人关系记录分析助手。请只依据下方证据分析“我”和“{session_name}”的互动，不推断不可观察的内心，不诊断人格或精神疾病，也不把消息数量直接等同于爱意。

请用中文 Markdown 输出《关系分析报告》，必须包含：
1. 摘要与数据边界
2. 关系阶段与时间范围
3. 双方主动性与投入变化
4. 分享欲、亲密或暧昧表达
5. 冲突与修复方式
6. 追逐—回避模式（证据不足就写无法判断）
7. 升温、降温时期与关键转折点
8. 可观察的对方态度变化
9. 我的关系行为模式
10. 证据台账

每个重要判断必须标注“事实 / 高可信推断 / 可能解释 / 无法判断”，并至少引用一条“准确时间｜我/对方｜短原文或忠实转述”。不要出现被污染的旧昵称。若证据相互矛盾，要并列呈现。

以下是本机程序从完整时间线压缩得到的证据：

{evidence}
"""


def analyze_timeline(timeline_path: Path, config: dict[str, Any], output_name: str = "relationship-report.md") -> Path:
    validate_local_config(config)
    runtime = Path(str(config.get("local_llm_runtime", ""))).expanduser()
    model = Path(str(config.get("local_llm_model", ""))).expanduser()
    if not runtime.is_file():
        raise RuntimeError(f"本地分析运行时不存在：{runtime}。请先运行 setup.bat。")
    if not model.is_file():
        raise RuntimeError(f"本地关系模型不存在：{model}。请先运行 setup.bat。")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
    session_name = str(timeline.get("session") or timeline_path.parent.name)
    evidence = build_evidence_pack(timeline, int(config.get("analysis_evidence_max_chars", 60_000)))
    prompt = build_prompt(session_name, evidence)
    cache_dir = PROJECT_ROOT / "cache" / "local-analysis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prompt-", dir=cache_dir) as temp:
        prompt_path = Path(temp) / "prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        command = [
            str(runtime), "-m", str(model), "-f", str(prompt_path),
            "--ctx-size", str(int(config.get("analysis_context", 32768))),
            "-n", str(int(config.get("analysis_max_tokens", 4096))),
            "--threads", str(int(config.get("analysis_threads", 10))),
            "--temp", str(float(config.get("analysis_temperature", 0.35))),
            "--top-p", "0.85", "--no-display-prompt", "--simple-io", "--single-turn", "--no-warmup", "--log-disable",
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"本地关系模型运行失败（退出码 {result.returncode}）：\n{detail}")
    report_text = result.stdout.strip()
    heading = re.search(r"(?m)^# .*关系分析报告.*$", report_text)
    if heading:
        report_text = report_text[heading.start() :]
    report_text = re.split(r"(?m)^\[ Prompt:|^Exiting\.\.\.$", report_text, maxsplit=1)[0].rstrip()
    if len(report_text) < 20 or "#" not in report_text:
        raise RuntimeError("本地模型没有生成有效关系分析报告，原时间线未受影响。")
    output = timeline_path.parent / output_name
    output.write_text(report_text + "\n", encoding="utf-8")
    return output
