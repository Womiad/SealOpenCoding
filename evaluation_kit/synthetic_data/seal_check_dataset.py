#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_check_dataset.py — 檢核合成訪談資料集，並產出《埋點對照表.csv》與 SHA256SUMS.txt。

檢查項目（對應評估計畫書 §7.5）：
  1. 每個埋點關鍵句能在對應逐字稿中「逐字」找到，且只出現一次（行號自動計算）。
  2. 行數在範圍內（主檔 40–90 行；thin 檔 15–40 行）。
  3. 受訪者實質內容占比（主檔受訪者字數 ≥ 60%）。
  4. 受訪者發言不得出現分析詞彙（取捨、條件式、行為模式、研究機會、語證、埋點、codebook、sensitizing）。
  5. 每行以「訪員：」或「受訪者：」開頭（允許少量「標籤獨立成行」的粗糙化）。

用法：
  python seal_check_dataset.py --anchors anchors.json --dirs 逐字稿 ../test_kit/逐字稿 --out 埋點對照表.csv
任何檢查失敗會列出並以非零碼結束；全部通過才寫出 CSV 與 SHA256SUMS.txt。
只用 Python 標準函式庫。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

FORBIDDEN = ["取捨", "條件式", "行為模式", "研究機會", "語證", "埋點", "codebook", "sensitizing"]
LABEL = re.compile(r"^(訪員|受訪者)：(.*)$")


def load_transcript(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--dirs", nargs="+", required=True, help="逐字稿資料夾（可多個）")
    parser.add_argument("--out", default="埋點對照表.csv")
    parser.add_argument("--sums", default="SHA256SUMS.txt")
    args = parser.parse_args()

    anchors = json.loads(Path(args.anchors).read_text(encoding="utf-8-sig"))
    anchors.pop("_comment", None)
    files: dict[str, Path] = {}
    for d in args.dirs:
        for p in Path(d).glob("*.txt"):
            files[p.name] = p

    errors, rows, sums = [], [], []
    for name, entries in anchors.items():
        path = files.get(name)
        if path is None:
            errors.append(f"[{name}] 找不到檔案（搜尋範圍：{args.dirs}）")
            continue
        lines = load_transcript(path)
        text = "\n".join(lines)
        is_thin = name.startswith("thin")
        # test_kit 的 P01–P03 早於本規格，只驗證埋點（行數、占比、詞彙規則不回溯套用）。
        legacy = name.startswith("P0")

        # 2. line-count range
        n = len([l for l in lines if l.strip()])
        low, high = (15, 40) if is_thin else (40, 90)
        if not legacy and not low <= n <= high:
            errors.append(f"[{name}] 行數 {n} 不在 {low}–{high}")

        # 3/4/5. label format, respondent ratio, forbidden vocabulary
        resp_chars = total_chars = 0
        bare = 0
        pending = ""
        for i, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line:
                continue
            m = LABEL.match(line)
            if m:
                speaker, content = m.group(1), m.group(2)
                if not content:
                    pending = speaker
                    continue
                pending = ""
            elif pending:
                speaker, content = pending, line
                pending = ""
            else:
                speaker, content = "", line
                bare += 1
            total_chars += len(content)
            if speaker == "受訪者":
                resp_chars += len(content)
                if not legacy:
                    for w in FORBIDDEN:
                        if w in content:
                            errors.append(f"[{name}] 第 {i} 行受訪者發言含分析詞彙「{w}」：{content[:30]}…")
        if bare > 0 and not legacy:
            errors.append(f"[{name}] 有 {bare} 行沒有講者標籤也不接在獨立標籤之後")
        if not is_thin and not legacy and total_chars and resp_chars / total_chars < 0.60:
            errors.append(f"[{name}] 受訪者字數占比 {resp_chars/total_chars:.0%} < 60%")

        # 1. anchors verbatim + unique, with auto line numbers
        for pid, ptype, quote, expected, scoring in entries:
            count = text.count(quote)
            if count == 0:
                errors.append(f"[{name}] {pid} 關鍵句找不到：「{quote}」")
                continue
            if count > 1:
                errors.append(f"[{name}] {pid} 關鍵句出現 {count} 次（必須唯一）：「{quote}」")
                continue
            line_no = next(i for i, l in enumerate(lines, 1) if quote in l)
            rows.append({
                "埋點ID": pid, "逐字稿": name, "類型": ptype, "行號": line_no,
                "關鍵句（逐字）": quote, "預期 code 要點": expected, "給分說明": scoring,
            })
        sums.append((hashlib.sha256(path.read_bytes()).hexdigest(), name))

    # cross-file: anchor IDs unique
    ids = [r["埋點ID"] for r in rows]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"埋點ID 重複：{dup}")

    if errors:
        print(f"未通過（{len(errors)} 項）：")
        for e in errors:
            print("  ✗", e)
        return 1

    with Path(args.out).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["埋點ID", "逐字稿", "類型", "行號", "關鍵句（逐字）", "預期 code 要點", "給分說明"])
        writer.writeheader()
        writer.writerows(rows)
    with Path(args.sums).open("w", encoding="utf-8") as handle:
        for digest, name in sorted(sums, key=lambda x: x[1]):
            handle.write(f"{digest}  {name}\n")
    scored = sum(1 for r in rows if r["類型"] != "排除")
    print(f"全部通過：{len(anchors)} 份逐字稿、{len(rows)} 個埋點（計分 {scored}、排除項 {len(rows)-scored}）")
    print(f"已寫出 {args.out} 與 {args.sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
