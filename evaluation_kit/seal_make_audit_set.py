#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_make_audit_set.py — 為 Part C（驗證行為實驗）製作稽核版 CSV。

從一份 Seal Open Coding 結果 CSV 出發：
  版 A：套用 edits.json 的修改（把指定的 coding 列改成「有問題的 code」），其餘不動；
  版 B：與版 A 相同，但清空 why_this_code、rationale、score_reason、五個子分數與 analytic_score，
        讓 Seal Code Reader 只顯示語證與 code（Reader 對空白欄位會安全略過）。

用法：
  python seal_make_audit_set.py 結果.csv --edits edits.json --out-a 稽核A.csv --out-b 稽核B.csv [--keep 12]

edits.json：鍵是 coding 列的序號（從 1 起算，不含文本簡介列），值是要覆蓋的欄位。
  {"3": {"code": "…"}, "7": {"speaker": "訪員", "segment_id": "S0037", "quote_verbatim": "…", "evidence_quote": "…", "code": "…"}}
--keep N：只保留前 N 筆 coding 列（受試者只稽核這些），文本簡介與研究方向列照常保留。
只用 Python 標準函式庫；不修改原始檔。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HIDDEN_IN_B = ["why_this_code", "rationale", "score_reason", "analytic_score", "confidence",
               "research_relevance", "behavior_pattern", "evidence_strength", "opportunity_potential", "inference_risk"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--edits", required=True)
    parser.add_argument("--out-a", required=True)
    parser.add_argument("--out-b", required=True)
    parser.add_argument("--keep", type=int, default=0, help="只保留前 N 筆 coding 列（0 = 全部）")
    args = parser.parse_args()

    src = Path(args.csv_path)
    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    edits = json.loads(Path(args.edits).read_text(encoding="utf-8"))

    coding_indexes = [i for i, r in enumerate(rows) if r.get("row_type") == "coding"]
    if args.keep:
        drop = set(coding_indexes[args.keep:])
        rows = [r for i, r in enumerate(rows) if i not in drop]
        coding_indexes = [i for i, r in enumerate(rows) if r.get("row_type") == "coding"]
    if not coding_indexes:
        print("這份 CSV 沒有 coding 列", file=sys.stderr)
        return 1

    applied = []
    for key, changes in edits.items():
        n = int(key)
        if not 1 <= n <= len(coding_indexes):
            print(f"序號 {n} 超出範圍（共 {len(coding_indexes)} 筆 coding）", file=sys.stderr)
            return 1
        row = rows[coding_indexes[n - 1]]
        for field, value in changes.items():
            if field not in fieldnames:
                print(f"欄位不存在：{field}", file=sys.stderr)
                return 1
            row[field] = value
        # keep Reader's fallback fields consistent
        if "why_this_code" in changes and "rationale" in fieldnames and "rationale" not in changes:
            row["rationale"] = changes["why_this_code"]
        applied.append(n)

    def write(path: str, hide: bool) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                out = dict(r)
                if hide and out.get("row_type") == "coding":
                    for field in HIDDEN_IN_B:
                        if field in out:
                            out[field] = ""
                writer.writerow(out)

    write(args.out_a, hide=False)
    write(args.out_b, hide=True)
    print(f"已修改 coding 列：{applied}；共 {len(coding_indexes)} 筆 coding → {args.out_a}（顯示理由）、{args.out_b}（隱藏理由）")
    print("請把植入的序號與類型登錄到記錄表 PartC_稽核 分頁。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
