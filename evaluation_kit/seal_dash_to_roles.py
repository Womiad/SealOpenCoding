#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_dash_to_roles.py — 把 Seal STT 的「破折號講者格式」轉成 Seal Open Coding 研究指引所預期的「訪員：／受訪者：」格式。

Seal STT 的 *_speakers.txt 與 *_polished_anonymized.txt 用這種寫法：
    講者1 的句子（行首沒有符號）
    - 講者2 的句子（行首一個「-」）
    -- 講者3 的句子
Seal Open Coding 會把「-」開頭視為通用講者標記，不會自動當成受訪者；
加上明確的角色標籤後，訪員的提問才會被正確排除、受訪者的話才會成為主句。

用法：
    python seal_dash_to_roles.py 輸入.txt 輸出.txt            # 講者1 → 訪員、講者2 → 受訪者（訪談通常訪員先開口）
    python seal_dash_to_roles.py 輸入.txt 輸出.txt --swap     # 反過來：講者1 → 受訪者、講者2 → 訪員
第三位以上的講者會標成「講者3：」，請自行改名。只用 Python 標準函式庫；不修改輸入檔。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DASH = re.compile(r"^\s*(-+)\s*(.*)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--swap", action="store_true", help="講者1 是受訪者、講者2 是訪員時使用")
    args = parser.parse_args()
    roles = {1: "訪員", 2: "受訪者"} if not args.swap else {1: "受訪者", 2: "訪員"}
    text = Path(args.src).read_text(encoding="utf-8-sig")
    out, counts = [], {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        m = DASH.match(line)
        number = len(m.group(1)) + 1 if m else 1
        content = m.group(2).strip() if m else line
        if not content:
            continue
        role = roles.get(number, f"講者{number}")
        counts[role] = counts.get(role, 0) + 1
        out.append(f"{role}：{content}")
    Path(args.dst).write_text("\n".join(out) + "\n", encoding="utf-8-sig")
    print("已寫入", args.dst, "｜各角色行數：", ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("請打開輸出檔確認「訪員」真的是提問的人；若相反，加 --swap 再跑一次。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
