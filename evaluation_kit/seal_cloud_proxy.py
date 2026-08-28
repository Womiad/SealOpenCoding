#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_cloud_proxy.py — 在 127.0.0.1 上模仿 Ollama API、把請求轉送到雲端（OpenAI 相容）API 的代理。

用途：Part B「本機 vs 雲端」比較。Seal Open Coding 只接受 loopback 位址，這個代理讓 Seal 不用改碼、
用同一套 prompt／分批／門檻，只換模型。**只能送合成資料**：啟動時必須設定 SEAL_EVAL_SYNTHETIC_ONLY=1
以示確認；代理本身不會檢查內容，責任在執行者。

Seal 會呼叫：
  GET  /api/tags   → 回 {"models": [{"name": "<別名>"}]}（V1.7.1 開始前檢查模型是否存在）
  POST /api/chat   → {"model", "messages", "stream": false, "format": "json", "options": {"temperature", "num_ctx"}}
                     期待回 {"message": {"role": "assistant", "content": "<JSON 字串>"}, "done": true}

啟動範例（PowerShell）：
  $env:SEAL_EVAL_SYNTHETIC_ONLY = "1"
  $env:SEAL_CLOUD_API_KEY = "sk-..."
  python seal_cloud_proxy.py --upstream https://api.openai.com/v1 --alias cloud-a=gpt-5 --port 11435

然後在 Seal Open Coding 的 GUI 把 Ollama 位址改成 http://127.0.0.1:11435、模型填 cloud-a。
Gemini 的 OpenAI 相容端點：--upstream https://generativelanguage.googleapis.com/v1beta/openai
Anthropic 的 OpenAI 相容端點：--upstream https://api.anthropic.com/v1（詳見各家文件；模型名以當時文件為準）

log 只記錄時間、模型、延遲、token 數與狀態，不記錄任何內容。
只用 Python 標準函式庫。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG = {
    "upstream": "",
    "api_key": "",
    "aliases": {},          # alias -> upstream model name
    "json_mode": True,      # send response_format={"type":"json_object"}
    "timeout": 600,
    "log_path": "",
    "max_tokens": 0,        # 0 = don't send
    "extra_headers": {},
}


def log_row(row: dict) -> None:
    path = CONFIG["log_path"]
    if not path:
        return
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "alias", "upstream_model", "status", "latency_s",
                                                    "prompt_tokens", "completion_tokens", "temperature", "num_ctx"])
        if new:
            writer.writeheader()
        writer.writerow(row)


def call_upstream(alias: str, messages: list, temperature: float) -> tuple[str, dict]:
    model = CONFIG["aliases"].get(alias)
    if model is None:
        raise KeyError(alias)
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if CONFIG["json_mode"]:
        payload["response_format"] = {"type": "json_object"}
    if CONFIG["max_tokens"]:
        payload["max_tokens"] = CONFIG["max_tokens"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {CONFIG['api_key']}"}
    headers.update(CONFIG["extra_headers"])
    request = urllib.request.Request(
        CONFIG["upstream"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=CONFIG["timeout"]) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"] or ""
    usage = body.get("usage", {}) or {}
    return content, usage


class Handler(BaseHTTPRequestHandler):
    server_version = "SealCloudProxy/1.0"

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quiet: never print request bodies
        sys.stderr.write("%s %s\n" % (datetime.now().strftime("%H:%M:%S"), fmt % args))

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            models = [{"name": alias, "model": alias, "details": {"family": "cloud-proxy"}} for alias in CONFIG["aliases"]]
            return self._send(200, {"models": models})
        if self.path == "/" or self.path.startswith("/api/version"):
            return self._send(200, {"version": "seal-cloud-proxy"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            return self._send(404, {"error": "only /api/chat is proxied"})
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        alias = str(req.get("model", ""))
        if alias not in CONFIG["aliases"]:
            return self._send(404, {"error": f"model '{alias}' not found; known: {list(CONFIG['aliases'])}"})
        if req.get("stream"):
            return self._send(400, {"error": "streaming not supported by this proxy"})
        options = req.get("options", {}) or {}
        temperature = float(options.get("temperature", 0.1))
        started = time.time()
        try:
            content, usage = call_upstream(alias, req.get("messages", []), temperature)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            log_row({"time": datetime.now().isoformat(timespec="seconds"), "alias": alias, "upstream_model": CONFIG["aliases"][alias],
                     "status": f"HTTP {exc.code}", "latency_s": round(time.time() - started, 1), "prompt_tokens": "", "completion_tokens": "",
                     "temperature": temperature, "num_ctx": options.get("num_ctx", "")})
            # 5xx/429 → 504 so Seal treats it as a timeout and bisects/retries instead of aborting the run
            code = 504 if exc.code >= 500 or exc.code == 429 else exc.code
            return self._send(code, {"error": f"upstream HTTP {exc.code}: {detail}"})
        except Exception as exc:  # noqa: BLE001
            log_row({"time": datetime.now().isoformat(timespec="seconds"), "alias": alias, "upstream_model": CONFIG["aliases"][alias],
                     "status": f"error {type(exc).__name__}", "latency_s": round(time.time() - started, 1), "prompt_tokens": "",
                     "completion_tokens": "", "temperature": temperature, "num_ctx": options.get("num_ctx", "")})
            return self._send(504, {"error": f"upstream error: {exc}"})
        latency = time.time() - started
        log_row({"time": datetime.now().isoformat(timespec="seconds"), "alias": alias, "upstream_model": CONFIG["aliases"][alias],
                 "status": "ok", "latency_s": round(latency, 1), "prompt_tokens": usage.get("prompt_tokens", ""),
                 "completion_tokens": usage.get("completion_tokens", ""), "temperature": temperature, "num_ctx": options.get("num_ctx", "")})
        return self._send(200, {
            "model": alias, "created_at": datetime.now().isoformat(),
            "message": {"role": "assistant", "content": content},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": usage.get("prompt_tokens", 0), "eval_count": usage.get("completion_tokens", 0),
            "total_duration": int(latency * 1e9),
        })


def main() -> int:
    parser = argparse.ArgumentParser(description="Loopback Ollama-compatible proxy to a cloud OpenAI-compatible API (synthetic data only).")
    parser.add_argument("--upstream", required=True, help="OpenAI 相容 base URL，例如 https://api.openai.com/v1")
    parser.add_argument("--alias", action="append", default=[], help="別名=上游模型名，可重複；例如 --alias cloud-a=gpt-5")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--api-key-env", default="SEAL_CLOUD_API_KEY", help="放 API key 的環境變數名稱")
    parser.add_argument("--no-json-mode", action="store_true", help="上游不支援 response_format 時使用")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--log", default="seal_cloud_proxy_log.csv", help="只記錄時間／token／延遲，不記錄內容")
    parser.add_argument("--header", action="append", default=[], help="額外 HTTP 標頭 key=value（例如 anthropic-version=2023-06-01）")
    args = parser.parse_args()

    if os.environ.get("SEAL_EVAL_SYNTHETIC_ONLY") != "1":
        print("拒絕啟動：這個代理會把逐字稿送到雲端。請先確認只會送「合成」資料，"
              "並設定環境變數 SEAL_EVAL_SYNTHETIC_ONLY=1。", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"找不到 API key：請設定環境變數 {args.api_key_env}", file=sys.stderr)
        return 2
    aliases = {}
    for item in args.alias:
        if "=" not in item:
            print(f"--alias 格式應為 別名=模型名：{item}", file=sys.stderr)
            return 2
        alias, model = item.split("=", 1)
        aliases[alias.strip()] = model.strip()
    if not aliases:
        print("至少要一個 --alias", file=sys.stderr)
        return 2
    CONFIG.update({"upstream": args.upstream, "api_key": key, "aliases": aliases, "json_mode": not args.no_json_mode,
                   "timeout": args.timeout, "log_path": args.log, "max_tokens": args.max_tokens,
                   "extra_headers": dict(h.split("=", 1) for h in args.header if "=" in h)})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Seal cloud proxy on http://127.0.0.1:{args.port}  upstream={args.upstream}  aliases={aliases}")
    print("在 Seal Open Coding 把 Ollama 位址改成上面的網址、模型填別名。Ctrl+C 結束。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
