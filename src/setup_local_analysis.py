from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import httpx


RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-win-cpu-x64.zip"
RUNTIME_SIZE = 18_560_055
RUNTIME_SHA256 = "14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1"
MODEL_URL = "https://www.modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/master/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_SIZE = 2_497_281_120
MODEL_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"
BASE = Path(r"D:\Codexxxx")
RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime" / "llama.cpp"
MODEL_DIR = BASE / "models" / "relationship-analysis"
MODEL_PATH = MODEL_DIR / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_size: int, expected_sha: str) -> Path:
    if destination.is_file() and destination.stat().st_size == expected_size:
        if sha256(destination) == expected_sha:
            print(f"已存在并通过校验：{destination}")
            return destination
        destination.unlink()
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with httpx.stream("GET", url, headers=headers, timeout=httpx.Timeout(60, connect=20), follow_redirects=True) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                done = offset if append else 0
                with part.open(mode) as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
                        done += len(chunk)
                        print(f"\r下载 {destination.name}：{done / 1024**2:.0f}/{expected_size / 1024**2:.0f} MB（{done * 100 / expected_size:.1f}%）", end="", flush=True)
            print()
            if part.stat().st_size != expected_size:
                raise RuntimeError(f"大小不符：{part.stat().st_size}/{expected_size}")
            if sha256(part) != expected_sha:
                part.unlink(missing_ok=True)
                raise RuntimeError("SHA-256 校验失败")
            part.replace(destination)
            return destination
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError(f"下载失败：{destination.name}：{exc}") from exc
            print(f"\n连接中断，保留已下载部分并重试（{attempt + 1}/5）……")
    raise AssertionError("unreachable")


def install_runtime() -> Path:
    executable = RUNTIME_DIR / "llama-cli.exe"
    if executable.is_file():
        print(f"本地分析运行时已存在：{executable}")
        return executable
    archive = Path(__file__).resolve().parents[1] / "cache" / "downloads" / "llama-b11146-bin-win-cpu-x64.zip"
    download(RUNTIME_URL, archive, RUNTIME_SIZE, RUNTIME_SHA256)
    staging = RUNTIME_DIR.with_name("llama.cpp-installing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as package:
        package.extractall(staging)
    found = next(staging.rglob("llama-cli.exe"), None)
    if found is None:
        raise RuntimeError("运行时压缩包中没有 llama-cli.exe。")
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    if found.parent == staging:
        staging.replace(RUNTIME_DIR)
    else:
        shutil.copytree(found.parent, RUNTIME_DIR)
        shutil.rmtree(staging)
    return executable


def main() -> int:
    runtime = install_runtime()
    model = download(MODEL_URL, MODEL_PATH, MODEL_SIZE, MODEL_SHA256)
    print(f"本地关系分析已就绪：\n运行时：{runtime}\n模型：{model}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消；下次运行会断点续传。")
        raise SystemExit(130)
