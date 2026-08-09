"""BioCLIP モデルを HuggingFace から事前ダウンロードしてローカルキャッシュに格納する。

HF が 504 等で不安定なときのために、失敗しても一定間隔でリトライし続ける。
一度キャッシュされれば api_cpu.py / api.py の起動は毎回のダウンロードを省ける
（HF_HUB_OFFLINE=1 を付ければ HF 障害中でも起動可能）。

使い方:
    .venv/bin/python prefetch_model.py
"""
from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "imageomics/bioclip-2.5-vith14"
MAX_ATTEMPTS = 0          # 0 = 無限リトライ（Ctrl-C で停止）
SLEEP_SECONDS = 30        # リトライ間隔


def main() -> int:
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[{attempt}] {REPO_ID} をダウンロード中 ...", flush=True)
            path = snapshot_download(repo_id=REPO_ID)
            print(f"完了: {path}", flush=True)
            return 0
        except (HfHubHTTPError, OSError, Exception) as e:  # noqa: BLE001
            print(f"[{attempt}] 失敗: {type(e).__name__}: {e}", flush=True)
            if MAX_ATTEMPTS and attempt >= MAX_ATTEMPTS:
                return 1
            print(f"{SLEEP_SECONDS}s 後に再試行します（Ctrl-C で中止）...", flush=True)
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中止しました。", flush=True)
        sys.exit(130)
