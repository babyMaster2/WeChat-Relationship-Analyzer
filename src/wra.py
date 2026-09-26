from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
OUTPUT_ROOT = PROJECT_ROOT / "output"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".silk", ".slk", ".aud"}


@dataclass
class Message:
    id: str
    timestamp: int
    time: str
    sender: str
    sender_id: str
    direction: str
    type: str
    text: str
    audio_path: str | None = None
    transcript: str | None = None
    transcript_status: str | None = None
    source: str | None = None


def load_config() -> dict[str, Any]:
    base = {
        "model": "medium",
        "language": "zh",
        "device": "auto",
        "cpu_compute_type": "int8",
        "gpu_compute_type": "float16",
        "beam_size": 5,
        "vad_filter": True,
        "local_only": True,
        "model_cache_dir": "../models/faster-whisper",
        "analysis_backend": "local",
    }
    if CONFIG_PATH.exists():
        base.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig")))
    if not base.get("local_only", True):
        raise RuntimeError("为保护隐私，本项目只允许 local_only=true。")
    return base


def resolve_model_cache_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("model_cache_dir") or "../models/faster-whisper")).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


def _download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    import httpx

    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            timeout = httpx.Timeout(60.0, connect=15.0)
            with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if not append:
                    offset = 0
                mode = "ab" if append else "wb"
                received = offset
                last_update = 0.0
                with part.open(mode) as stream:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= 1.0:
                            percent = received * 100 / expected_size if expected_size else 0
                            print(
                                f"\r下载 {destination.name}：{received / 1024 / 1024:.1f} / "
                                f"{expected_size / 1024 / 1024:.1f} MB（{percent:.1f}%）",
                                end="",
                                flush=True,
                            )
                            last_update = now
            if part.stat().st_size == expected_size:
                print(flush=True)
                part.replace(destination)
                return
            raise RuntimeError(f"文件大小不完整：{part.stat().st_size} / {expected_size}")
        except Exception:
            if attempt == 3:
                raise
            print(f"\n下载中断，正在断点续传（第 {attempt + 1}/3 次）……", flush=True)


def download_model_from_mirror(repo_id: str, target_dir: Path) -> Path:
    import httpx

    endpoint = "https://hf-mirror.com"
    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    known_checksums = {
        "Systran/faster-whisper-medium": {
            "model.bin": "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"
        }
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        url = f"{endpoint}/{repo_id}/resolve/main/{filename}"
        head = httpx.head(url, timeout=30.0, follow_redirects=True)
        head.raise_for_status()
        expected_size = int(head.headers["content-length"])
        destination = target_dir / filename
        if not destination.exists() or destination.stat().st_size != expected_size:
            _download_with_resume(url, destination, expected_size)
        expected_sha = known_checksums.get(repo_id, {}).get(filename)
        if expected_sha:
            print(f"校验 {filename}……", flush=True)
            hasher = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != expected_sha:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"{filename} 校验失败，请重新运行以重下该文件。")
    return target_dir


def download_model_from_modelscope(model: str, target_dir: Path) -> Path:
    import httpx

    repo_id = f"gpustack/faster-whisper-{model}"
    endpoint = "https://www.modelscope.cn/models"
    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    known_checksums = {
        "medium": {
            "model.bin": "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"
        }
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        url = f"{endpoint}/{repo_id}/resolve/master/{filename}"
        with httpx.stream(
            "GET",
            url,
            headers={"Range": "bytes=0-0"},
            timeout=httpx.Timeout(30.0, connect=15.0),
            follow_redirects=True,
        ) as probe:
            probe.raise_for_status()
            content_range = probe.headers.get("content-range", "")
            expected_size = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else int(probe.headers["content-length"])
        destination = target_dir / filename
        if not destination.exists() or destination.stat().st_size != expected_size:
            _download_with_resume(url, destination, expected_size)
        expected_sha = known_checksums.get(model, {}).get(filename)
        if expected_sha:
            print(f"校验 {filename}……", flush=True)
            hasher = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_sha:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"{filename} 校验失败，请重新运行以重下该文件。")
    return target_dir


def _model_dir_is_ready(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"))


def prepare_model(config: dict[str, Any]) -> Path:
    """Download the public Whisper model with visible progress before transcription starts."""
    from huggingface_hub import snapshot_download

    model = str(config["model"])
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path.resolve()

    repo_id = model if "/" in model else f"Systran/faster-whisper-{model}"
    cache_dir = resolve_model_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    if _model_dir_is_ready(local_dir):
        return local_dir
    print(f"首次准备语音模型：{model}（只需下载一次，保存在 D 盘）", flush=True)
    print("正在下载并校验模型；公开模型无需 HF_TOKEN，请保持网络连接……", flush=True)
    domestic_failure = ""
    for domestic_attempt in range(1, 4):
        try:
            downloaded = str(download_model_from_modelscope(model, local_dir))
            print(f"语音模型准备完成：{downloaded}", flush=True)
            return Path(downloaded)
        except Exception as domestic_error:
            domestic_failure = str(domestic_error)
            if domestic_attempt < 3:
                print(f"国内下载线路暂时中断，正在重试（第 {domestic_attempt + 1}/3 次）……", flush=True)
    print("国内下载线路不可用，正在尝试官方下载线路……", flush=True)
    official_failure = ""
    mirror_failure = ""
    try:
        downloaded = snapshot_download(
            repo_id,
            cache_dir=str(cache_dir),
            endpoint="https://huggingface.co",
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ],
        )
    except Exception as official_error:
        official_failure = str(official_error)
        print("官方下载线路连接失败，正在切换最后备用线路……", flush=True)
        downloaded = None
        for mirror_attempt in range(1, 4):
            try:
                downloaded = str(download_model_from_mirror(repo_id, local_dir))
                break
            except Exception as mirror_error:
                mirror_failure = str(mirror_error)
                if mirror_attempt < 3:
                    print(f"备用线路暂时中断，正在重试（第 {mirror_attempt + 1}/3 次）……", flush=True)
    if downloaded is None:
        raise RuntimeError(
            "语音模型的官方下载线路和备用线路都不可用。\n"
            f"模型缓存目录：{cache_dir}\n"
            "请检查网络后重新双击 start.bat；已下载部分会自动续传。\n"
            f"国内下载：{domestic_failure}\n官方下载：{official_failure}\n备用下载：{mirror_failure}"
        )
    print(f"语音模型准备完成：{downloaded}", flush=True)
    return Path(downloaded)


def sanitize_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return value[:80] or "unknown-contact"


def choose_input() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if messagebox.askyesno("微信聊天关系分析", "是否选择 WeFlow 导出的整个文件夹？\n选择“否”可直接选 JSON 文件。"):
            selected = filedialog.askdirectory(title="选择 WeFlow 导出文件夹")
        else:
            selected = filedialog.askopenfilename(title="选择 WeFlow JSON", filetypes=[("JSON", "*.json"), ("全部文件", "*.*")])
        root.destroy()
        return Path(selected) if selected else None
    except Exception:
        raw = input("请输入 WeFlow JSON 文件或导出文件夹路径：").strip().strip('"')
        return Path(raw) if raw else None


def find_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in {".json", ".jsonl"} else []
    files = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"} and "run-report" not in p.name and "timeline" not in p.name:
            files.append(p)
    return sorted(files)


def read_json(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_nested(obj: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def normalize_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = int(value)
        if number > 10_000_000_000:
            number //= 1000
        return number
    text = str(value or "").strip()
    if text.isdigit():
        return normalize_timestamp(int(text))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    return 0


def format_local_time(timestamp: int, fallback: str = "") -> str:
    if timestamp > 0:
        return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return fallback or "未知时间"


def flatten_messages(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if isinstance(data, dict):
        meta = data
        for key in ("messages", "messageList", "records", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)], meta
    if isinstance(data, list):
        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("_type") == "message" or any(k in item for k in ("createTime", "timestamp", "content", "text")):
                rows.append(item)
        return rows, meta
    return [], meta


def build_audio_index(root: Path) -> tuple[dict[str, Path], list[Path]]:
    base = root if root.is_dir() else root.parent
    audio = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    index: dict[str, Path] = {}
    for path in audio:
        for token in re.findall(r"[A-Za-z0-9_-]{4,}", path.stem):
            index.setdefault(token.lower(), path)
        index.setdefault(path.name.lower(), path)
    return index, audio


def resolve_audio_path(raw: dict[str, Any], json_path: Path, content: str, audio_index: dict[str, Path], audio_files: list[Path]) -> Path | None:
    candidates = [
        raw.get("audio_path"), raw.get("audioPath"), raw.get("voicePath"), raw.get("mediaPath"),
        raw.get("filePath"), raw.get("path"), raw.get("src"), content,
    ]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        cleaned = candidate.strip().replace("/", os.sep)
        p = Path(cleaned)
        if not p.is_absolute():
            p = (json_path.parent / p).resolve()
        if p.exists() and p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            return p

    tokens = []
    for key in ("platformMessageId", "msgId", "localId", "serverId", "svrId"):
        value = raw.get(key)
        if value not in (None, ""):
            tokens.append(str(value).lower())
    for token in tokens:
        if token in audio_index:
            return audio_index[token]
        for key, path in audio_index.items():
            if token in key:
                return path
    if len(audio_files) == 1:
        return audio_files[0]
    return None


def infer_sender_roles(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, str]:
    """Infer stable sender-id roles from media XML when an exporter corrupts isSend/nicknames."""
    session = meta.get("session") if isinstance(meta.get("session"), dict) else {}
    contact_id = str(get_nested(session, ("id", "wxid", "username"), ""))
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        sender_id = str(get_nested(row, ("sender_id", "senderId", "senderUsername"), ""))
        content = str(row.get("content") or row.get("text") or "")
        match = re.search(r'fromusername\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
        if sender_id and match and contact_id:
            votes[sender_id]["received" if match.group(1) == contact_id else "sent"] += 1
    roles: dict[str, str] = {}
    for sender_id, counts in votes.items():
        if counts:
            role, count = counts.most_common(1)[0]
            if count >= 1 and count > counts["sent" if role == "received" else "received"]:
                roles[sender_id] = role
    return roles


def normalize_message(
    raw: dict[str, Any], json_path: Path, audio_index: dict[str, Path], audio_files: list[Path],
    self_name: str, sender_roles: dict[str, str] | None = None,
) -> Message:
    timestamp = normalize_timestamp(get_nested(raw, ("createTime", "timestamp", "time", "create_time", "date"), 0))
    fallback_time = str(get_nested(raw, ("formattedTime", "formatted_time", "timeText"), ""))
    sender_id = str(get_nested(raw, ("senderUsername", "sender_id", "senderId", "fromUser", "wxid"), ""))
    is_send = get_nested(raw, ("isSend", "is_send", "fromMe", "isSelf"), None)
    direction = (sender_roles or {}).get(sender_id)
    if direction not in {"sent", "received"}:
        direction = "sent" if str(is_send).lower() in {"1", "true", "yes"} or sender_id == "2" else "received"
    sender = "我" if direction == "sent" else "对方"
    msg_type = str(get_nested(raw, ("type", "messageType", "msg_type", "localType"), "text"))
    local_type = str(raw.get("localType", ""))
    content = str(get_nested(raw, ("content", "text", "message", "body"), "") or "")
    voice = local_type == "34" or "voice" in msg_type.lower() or "语音" in msg_type or Path(content.replace("/", os.sep)).suffix.lower() in AUDIO_EXTENSIONS
    audio_path = resolve_audio_path(raw, json_path, content, audio_index, audio_files) if voice else None
    transcript = get_nested(raw, ("transcript", "voiceText", "voiceTranscript"), None)
    if isinstance(transcript, str):
        transcript = re.sub(r"^\[语音转文字\]\s*", "", transcript).strip() or None
    text = "" if voice and audio_path and content.replace("/", os.sep).lower().endswith(tuple(AUDIO_EXTENSIONS)) else content
    msg_id = str(get_nested(raw, ("platformMessageId", "msgId", "localId", "id", "serverId"), ""))
    if not msg_id:
        msg_id = hashlib.sha1(f"{timestamp}|{sender_id}|{content}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    return Message(
        id=msg_id,
        timestamp=timestamp,
        time=format_local_time(timestamp, fallback_time),
        sender=sender,
        sender_id=sender_id,
        direction=direction,
        type="voice" if voice else msg_type,
        text=text,
        audio_path=str(audio_path) if audio_path else None,
        transcript=transcript,
        transcript_status="provided" if transcript else ("pending" if audio_path else ("missing-audio" if voice else None)),
        source=str(json_path),
    )


def decode_silk_to_wav(source: Path, temp_dir: Path) -> Path:
    try:
        import pilk_nogil  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 pilk-nogil，无法解码 SILK") from exc
    wav_path = temp_dir / f"{source.stem}.wav"
    for attempt in range(2):
        temp_dir.mkdir(parents=True, exist_ok=True)
        wav_path.unlink(missing_ok=True)
        pilk_nogil.silk_to_wav(str(source), str(wav_path), rate=24000)
        if wav_path.exists() and wav_path.stat().st_size > 44:
            return wav_path
        if attempt == 0:
            time.sleep(0.05)
    raise RuntimeError(f"SILK 解码器未生成有效 WAV：{source.name}")


class LocalTranscriber:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.device = "not-loaded"
        self.compute_type = ""
        self.fallback_reason: str | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        model_path = prepare_model(self.config)
        requested = str(self.config.get("device", "auto"))
        if requested in {"auto", "cuda"}:
            try:
                self.model = WhisperModel(str(model_path), device="cuda", compute_type=str(self.config["gpu_compute_type"]), local_files_only=True)
                self.device = "cuda"
                self.compute_type = str(self.config["gpu_compute_type"])
                return
            except Exception as exc:
                self.fallback_reason = f"GPU 不可用，已回退 CPU：{exc}"
                if requested == "cuda":
                    print(self.fallback_reason)
        self.model = WhisperModel(str(model_path), device="cpu", compute_type=str(self.config["cpu_compute_type"]), local_files_only=True)
        self.device = "cpu"
        self.compute_type = str(self.config["cpu_compute_type"])

    def transcribe(self, source: Path, temp_dir: Path) -> str:
        self._load()
        actual = decode_silk_to_wav(source, temp_dir) if source.suffix.lower() in {".silk", ".slk", ".aud"} else source
        segments, _ = self.model.transcribe(
            str(actual),
            language=str(self.config.get("language") or "zh"),
            beam_size=int(self.config.get("beam_size", 5)),
            vad_filter=bool(self.config.get("vad_filter", True)),
        )
        return "".join(segment.text for segment in segments).strip()


def get_session_name(meta: dict[str, Any], json_path: Path) -> tuple[str, str]:
    session = meta.get("session") if isinstance(meta.get("session"), dict) else {}
    name = str(get_nested(session, ("displayName", "name", "nickname", "remark"), "") or get_nested(meta, ("contact", "name", "title"), "") or json_path.stem)
    self_info = meta.get("self") if isinstance(meta.get("self"), dict) else {}
    self_name = str(get_nested(self_info, ("displayName", "nickname", "name"), "我"))
    return name, self_name


def _message_markdown(message: Message) -> str:
    body = message.transcript if message.type == "voice" and message.transcript else message.text
    if message.type == "voice":
        status = "已转写" if message.transcript else "未转写"
        body = f"[语音｜{status}] {body or ''}".rstrip()
    return "\n".join([f"## {message.time}｜{message.sender}", "", body or f"[{message.type}]", ""])


def write_markdown_timeline(
    target: Path,
    session_name: str,
    messages: list[Message],
    max_part_bytes: int = 1_000_000,
    obsidian_links: bool = False,
) -> list[Path]:
    """Write a small index plus size-bounded monthly Markdown parts."""
    target.mkdir(parents=True, exist_ok=True)
    parts_dir = target / "timeline-parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[Message]] = {}
    for message in messages:
        month = message.time[:7] if re.fullmatch(r"\d{4}-\d{2}", message.time[:7]) else "未知日期"
        grouped.setdefault(month, []).append(message)

    created: list[Path] = []
    index_rows: list[tuple[str, str, int, str, str]] = []
    for month, month_messages in grouped.items():
        chunks: list[list[Message]] = []
        current: list[Message] = []
        current_size = 0
        for message in month_messages:
            block_size = len(_message_markdown(message).encode("utf-8"))
            if current and current_size + block_size > max_part_bytes:
                chunks.append(current)
                current = []
                current_size = 0
            current.append(message)
            current_size += block_size
        if current:
            chunks.append(current)

        for part_number, chunk in enumerate(chunks, 1):
            suffix = "" if part_number == 1 else f"-{part_number:02d}"
            filename = f"{month}{suffix}.md"
            path = parts_dir / filename
            header = [
                "---",
                f'contact: "{session_name}"',
                f'period: "{month}"',
                "local_only: true",
                "---",
                "",
                "> [!info] 本页由微信聊天关系分析器在本地生成。语音转写可能有误，重要判断需回看上下文。",
                "",
            ]
            path.write_text("\n".join(header + [_message_markdown(m) for m in chunk]), encoding="utf-8")
            created.append(path)
            label = month if len(chunks) == 1 else f"{month}（{part_number}/{len(chunks)}）"
            if obsidian_links:
                relative_note = path.relative_to(target).with_suffix("").as_posix()
                link = f"[[{relative_note}|{label}]]"
            else:
                link = f"[{label}](timeline-parts/{filename})"
            index_rows.append((link, label, len(chunk), chunk[0].time, chunk[-1].time))

    index = [
        "---",
        f'contact: "{session_name}"',
        "type: wechat-timeline-index",
        "local_only: true",
        "---",
        "",
        "> [!summary] 浏览说明",
        "> 完整聊天已按月份和文件大小拆分。请从下方目录进入；完整结构化数据仍保存在项目的 `timeline.json`。",
        "",
        f"消息总数：{len(messages)}",
        "",
        "| 时间线 | 消息数 | 起始 | 结束 |",
        "|---|---:|---|---|",
    ]
    index.extend(f"| {link} | {count} | {start} | {end} |" for link, _label, count, start, end in index_rows)
    index.extend(["", "> [!warning] 隐私", "> 本目录含私人聊天内容，仅在本机使用，不要上传到公开服务。", ""])
    (target / "timeline.md").write_text("\n".join(index), encoding="utf-8")
    return created


def reuse_existing_transcripts(messages: list[Message], timeline_path: Path) -> int:
    if not timeline_path.exists():
        return 0
    try:
        data = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return 0
    previous = {
        (str(item.get("id", "")), int(item.get("timestamp", 0))): item
        for item in data.get("messages", [])
        if isinstance(item, dict) and item.get("transcript")
    }
    reused = 0
    for message in messages:
        item = previous.get((message.id, message.timestamp))
        if item and not message.transcript:
            message.transcript = str(item["transcript"])
            message.transcript_status = str(item.get("transcript_status") or "ok")
            reused += 1
    return reused


def write_timeline_json(target: Path, session_name: str, messages: list[Message]) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "timeline.json"
    pending = target / "timeline.json.tmp"
    payload = {
        "schema": "wechat-relationship-timeline/v1",
        "session": session_name,
        "generatedAt": datetime.now().astimezone().isoformat(),
        "localOnly": True,
        "messages": [asdict(message) for message in messages],
    }
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(destination)
    return destination


def write_outputs(session_name: str, messages: list[Message], report: dict[str, Any]) -> Path:
    target = OUTPUT_ROOT / sanitize_name(session_name)
    target.mkdir(parents=True, exist_ok=True)
    write_timeline_json(target, session_name, messages)

    write_markdown_timeline(target, session_name, messages)

    obsidian_root = str(load_config().get("obsidian_output_dir") or "").strip()
    if obsidian_root:
        obsidian_target = Path(obsidian_root).expanduser() / sanitize_name(session_name)
        write_markdown_timeline(obsidian_target, session_name, messages, obsidian_links=True)

    request = f"""# 分析请求\n\n请使用 `$relationship-analysis` 分析本目录中的 `timeline.json`（联系人：{session_name}）。\n\n要求覆盖关系阶段、主动性与投入变化、分享欲、亲密或暧昧表达、冲突与修复、追逐—回避模式、升温和降温时期、关键转折点；所有重要判断引用具体时间、发送者和原文证据，并严格标注“事实 / 高可信推断 / 可能解释 / 无法判断”。不得诊断人格或精神疾病，不把消息数量直接等同于爱意。\n"""
    (target / "analysis-request.md").write_text(request, encoding="utf-8")
    (target / "run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def process(input_path: Path, skip_transcription: bool = False) -> int:
    config = load_config()
    json_files = find_json_files(input_path)
    if not json_files:
        raise RuntimeError("未找到 WeFlow JSON/JSONL 文件。请在 WeFlow 中选择 JSON 并勾选导出语音。")
    audio_index, audio_files = build_audio_index(input_path)
    transcriber = LocalTranscriber(config)
    grouped: dict[str, list[Message]] = {}
    errors: list[dict[str, str]] = []

    for json_path in json_files:
        try:
            data = read_json(json_path)
            rows, meta = flatten_messages(data)
            if not rows:
                continue
            session_name, self_name = get_session_name(meta, json_path)
            sender_roles = infer_sender_roles(rows, meta)
            grouped.setdefault(session_name, [])
            grouped[session_name].extend(normalize_message(row, json_path, audio_index, audio_files, self_name, sender_roles) for row in rows)
        except Exception as exc:
            errors.append({"file": str(json_path), "error": str(exc)})

    if not grouped:
        raise RuntimeError("JSON 文件存在，但没有识别到消息数组。")

    audio_work_root = PROJECT_ROOT / "cache" / "audio-work"
    audio_work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wra-audio-", dir=audio_work_root) as tmp:
        temp_dir = Path(tmp)
        for session_name, messages in grouped.items():
            messages.sort(key=lambda m: (m.timestamp, m.id))
            seen: set[str] = set()
            unique = []
            for message in messages:
                key = f"{message.id}|{message.timestamp}|{message.sender_id}"
                if key not in seen:
                    seen.add(key)
                    unique.append(message)
            messages[:] = unique

            existing_timeline = OUTPUT_ROOT / sanitize_name(session_name) / "timeline.json"
            reused = reuse_existing_transcripts(messages, existing_timeline)
            if reused:
                print(f"[{session_name}] 已复用 {reused} 条现有语音转写。")

            if not skip_transcription:
                pending = [m for m in messages if m.type == "voice" and m.audio_path and not m.transcript]
                for i, message in enumerate(pending, 1):
                    print(f"[{session_name}] 转写语音 {i}/{len(pending)}：{Path(message.audio_path).name}")
                    try:
                        message.transcript = transcriber.transcribe(Path(message.audio_path), temp_dir)
                        message.transcript_status = "ok" if message.transcript else "empty"
                    except Exception as exc:
                        message.transcript_status = "failed"
                        errors.append({"file": message.audio_path, "error": str(exc)})
                    if i % 10 == 0 or i == len(pending):
                        checkpoint_target = OUTPUT_ROOT / sanitize_name(session_name)
                        write_timeline_json(checkpoint_target, session_name, messages)
                        print(f"[{session_name}] 已保存断点：{i}/{len(pending)}")

            report = {
                "session": session_name,
                "localOnly": True,
                "messages": len(messages),
                "voiceMessages": sum(1 for m in messages if m.type == "voice"),
                "transcribedVoices": sum(1 for m in messages if m.type == "voice" and m.transcript),
                "missingVoiceFiles": sum(1 for m in messages if m.type == "voice" and not m.audio_path and not m.transcript),
                "model": config["model"],
                "device": transcriber.device,
                "computeType": transcriber.compute_type,
                "gpuFallbackReason": transcriber.fallback_reason,
                "errors": errors,
            }
            target = write_outputs(session_name, messages, report)
            print(f"已生成：{target}")
            if config.get("analysis_backend", "local") == "local":
                from local_analysis import analyze_timeline

                print(f"[{session_name}] 正在用本地模型生成关系分析报告……")
                analysis_path = analyze_timeline(target / "timeline.json", config)
                obsidian_root = str(config.get("obsidian_output_dir") or "").strip()
                if obsidian_root:
                    obsidian_report = Path(obsidian_root).expanduser() / sanitize_name(session_name) / "relationship-report.md"
                    obsidian_report.parent.mkdir(parents=True, exist_ok=True)
                    obsidian_report.write_text(analysis_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[{session_name}] 关系分析已生成：{analysis_path}")
    return 0


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="wra-test-") as tmp:
        root = Path(tmp)
        fixture = {
            "session": {"displayName": "最小测试联系人"},
            "messages": [
                {"localId": 1, "createTime": 1700000000, "localType": 1, "content": "你好", "isSend": 1, "senderDisplayName": "我"},
                {"localId": 2, "createTime": 1700000010, "localType": 1, "content": "你好呀", "isSend": 0, "senderDisplayName": "对方"},
            ],
        }
        test_json = root / "fixture.json"
        test_json.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        data = read_json(test_json)
        rows, meta = flatten_messages(data)
        assert len(rows) == 2
        name, self_name = get_session_name(meta, test_json)
        index, audio = build_audio_index(root)
        normalized = [normalize_message(row, test_json, index, audio, self_name, infer_sender_roles(rows, meta)) for row in rows]
        assert name == "最小测试联系人" and normalized[0].text == "你好"
        import faster_whisper  # noqa: F401
        try:
            import pilk_nogil  # noqa: F401
        except ImportError:
            print("提示：SILK 可选解码器未安装；WeFlow WAV 语音不受影响。")
    print("自检通过：JSON 解析、时间线规范化和 faster-whisper 依赖均可用。")
    return 0


def check() -> int:
    config = load_config()
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"Python：{sys.version.split()[0]}")
    print(f"Whisper 模型：{config['model']}")
    print(f"模型缓存：{resolve_model_cache_dir(config)}")
    try:
        import faster_whisper  # noqa: F401
        print("faster-whisper：已安装")
    except Exception as exc:
        print(f"faster-whisper：不可用（{exc}）")
        return 1
    try:
        import pilk_nogil  # noqa: F401
        print("SILK 解码：已安装")
    except Exception as exc:
        print(f"SILK 解码：不可用（{exc}）；WeFlow WAV 不受影响")
    print("NVIDIA：运行语音转写时自动探测；失败自动回退 CPU。")
    print("隐私模式：全本地，不调用云端转写 API。")
    try:
        from local_analysis import validate_local_config

        validate_local_config(config)
        runtime = Path(str(config.get("local_llm_runtime", "")))
        model = Path(str(config.get("local_llm_model", "")))
        print(f"本地关系分析：运行时{'已就绪' if runtime.is_file() else '缺失'}；模型{'已就绪' if model.is_file() else '缺失'}")
    except Exception as exc:
        print(f"本地关系分析：不可用（{exc}）")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="WeFlow 微信聊天时间线与本地语音转写")
    parser.add_argument("input", nargs="?", help="WeFlow JSON 文件或导出目录")
    parser.add_argument("--skip-transcription", action="store_true", help="只合并时间线，不运行 Whisper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--analyze-output", help="仅分析已有 timeline.json，不重新转写")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.check:
        return check()
    if args.analyze_output:
        from local_analysis import analyze_timeline

        result = analyze_timeline(Path(args.analyze_output).expanduser().resolve(), load_config())
        print(f"关系分析已生成：{result}")
        return 0
    input_path = Path(args.input).expanduser().resolve() if args.input else choose_input()
    if input_path is None:
        print("已取消。")
        return 0
    if not input_path.exists():
        raise FileNotFoundError(f"路径不存在：{input_path}")
    return process(input_path, skip_transcription=args.skip_transcription)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已取消。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{exc}")
        if os.environ.get("WRA_DEBUG") == "1":
            traceback.print_exc()
        raise SystemExit(1)
