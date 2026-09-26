from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = PROJECT_ROOT / "input"


def safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return value[:80] or "wechat-contact"


def main() -> int:
    try:
        from wechatauto import MediaDownloader, WeChatDB
    except ImportError as exc:
        raise RuntimeError("尚未安装 wechatauto-replica，请先运行 setup.bat。") from exc

    print("正在连接已登录的 Windows 微信 4.x，并只读扫描本地数据库密钥……")
    db = WeChatDB()
    keyword = input("请输入联系人昵称、备注或 wxid 关键字：").strip()
    if not keyword:
        print("未输入联系人，已取消。")
        return 0
    matches = db.search_contact(keyword)
    if not matches:
        print("没有找到联系人。请确认微信已登录，并尝试更短的关键字。")
        return 1

    print("\n匹配结果：")
    for i, item in enumerate(matches[:20], 1):
        username = str(item.get("username") or item.get("userName") or "")
        display = str(item.get("remark") or item.get("nick_name") or item.get("nickname") or item.get("nickName") or username)
        print(f"{i}. {display} ({username})")
    choice = input("请输入序号：").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= min(len(matches), 20)):
        print("序号无效，已取消。")
        return 1

    selected = matches[int(choice) - 1]
    username = str(selected.get("username") or selected.get("userName") or "")
    display = str(selected.get("remark") or selected.get("nick_name") or selected.get("nickname") or selected.get("nickName") or username)
    target = EXPORT_ROOT / f"{safe_name(display)}-{datetime.now():%Y%m%d-%H%M%S}"
    voices = target / "voices"
    voices.mkdir(parents=True, exist_ok=True)

    print("正在读取完整历史记录；聊天较长时请耐心等待……")
    messages = db.get_messages(username, limit=1_000_000)
    downloader = MediaDownloader(db)
    exported = []
    voice_ok = 0
    voice_failed = 0
    for idx, message in enumerate(messages, 1):
        row = dict(message)
        type_name = str(row.get("type") or "")
        local_type_raw = row.get("local_type") or row.get("localType") or 0
        try:
            local_type = int(local_type_raw)
        except (TypeError, ValueError):
            local_type = 0
        sender_username = str(row.get("sender_username") or "")
        is_send = str(row.get("sender_id") or "") == "2"
        row["isSend"] = 1 if is_send else 0
        row["senderDisplayName"] = "我" if is_send else (db.get_nickname(sender_username) if sender_username else "对方")
        if local_type == 34 or type_name == "语音" or type_name.lower() == "voice":
            try:
                local_id = row.get("local_id") or row.get("localId")
                downloaded = downloader.download_voice(username, int(local_id), str(voices))
                if downloaded:
                    source = Path(downloaded)
                    destination = voices / source.name
                    if source.resolve() != destination.resolve():
                        destination.write_bytes(source.read_bytes())
                    row["audioPath"] = str(destination.relative_to(target)).replace("\\", "/")
                    voice_ok += 1
                else:
                    voice_failed += 1
            except Exception as exc:
                row["voiceExportError"] = str(exc)
                voice_failed += 1
        exported.append(row)
        if idx % 1000 == 0:
            print(f"已处理 {idx}/{len(messages)} 条消息")

    payload = {
        "exporter": "wechatauto-replica",
        "session": {"id": username, "displayName": display},
        "messages": exported,
        "voiceExport": {"success": voice_ok, "failed": voice_failed},
    }
    output = target / "messages.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"导出完成：{target}")
    print(f"消息 {len(exported)} 条；语音成功 {voice_ok}，失败 {voice_failed}。")
    print("下一步双击 start.bat，并选择这个导出文件夹。")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"导出失败：{exc}")
        print("请确认微信 4.x 已登录；首次读取可能需要以管理员身份运行 export-chat.bat。")
        raise SystemExit(1)
