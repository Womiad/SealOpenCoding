#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_eval_collect.py — 收集 Part B 的 Seal Open Coding 結果 CSV，輸出可貼進《Seal 評估記錄表.xlsx》的表格。

預期資料夾結構（條件／run／CSV）：
  results/
    L2/run1/P01_個人創作者.csv
    L2/run2/...
    C1/run1/...
用法：
  python seal_eval_collect.py results/ --key 埋點對照表.csv --out collected/
輸出（都是 UTF-8 with BOM，Excel 可直接開）：
  collected/summary.csv            每份 CSV 一列：條件、run、逐字稿、code 列數、code_count_explanation 解析出的計數、方向數、平均分數、訪員主句數
  collected/codes_long.csv         每筆 code 一列（貼進 B_人工評分 的 A–J 欄）
  collected/planted_autocheck.csv  埋點 × 條件 × run 的自動比對（關鍵句是否出現在任一 code 的語證／上下文中），貼進 B_埋點評分 後人工修正
  collected/stability.csv          同條件 3 次 run 之間的 Jaccard（以主句 segment_id 集合近似）

埋點對照表 CSV 欄位（從記錄表 B_埋點對照表 另存）：埋點ID, 逐字稿, 類型, 行號, 關鍵句（逐字）, 預期 code 要點, ...
只用 Python 標準函式庫。
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import unicodedata
from itertools import combinations
from pathlib import Path

RE_INITIAL = re.compile(r"(\d+) 個可定位初始候選|共保留 (\d+) 個可定位的初始 code")
RE_ELIGIBLE = re.compile(r"(\d+) 個(?:候選)?通過(?:品質門檻|研究相關性)")
RE_FINAL = re.compile(r"(?:最後保留|聚焦後保留|只成功保留) (\d+) 個")
RE_SKIPPED = re.compile(r"有 (\d+) 個片段因持續逾時")
RE_MINIMUM = re.compile(r"設定至少 (\d+) 個")

PUNCT = "，。、；：？！「」『』（）()〔〕【】《》〈〉—…－-‐,.;:?!\"'“”‘’ 　\t"


def norm(text: str) -> str:
    """Normalise for matching: NFKC (full→half width), drop punctuation and spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if ch not in PUNCT and not ch.isspace())


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def first_int(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text or "")
    if not m:
        return ""
    return next((g for g in m.groups() if g), "")


def condition_and_run(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root).parts
    condition = rel[0] if len(rel) >= 3 else "?"
    run = rel[1] if len(rel) >= 3 else "?"
    m = re.search(r"(\d+)", run)
    return condition, (m.group(1) if m else run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="results 資料夾（條件/run/CSV）")
    parser.add_argument("--key", help="埋點對照表 CSV")
    parser.add_argument("--out", default="collected")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    csv_paths = sorted(p for p in root.rglob("*.csv") if not p.name.endswith(".partial"))
    if not csv_paths:
        print("找不到任何 CSV", file=sys.stderr)
        return 1

    summary, codes_long = [], []
    segsets: dict[tuple[str, str], dict[str, set]] = {}  # (condition, transcript) -> run -> set(segment_id)
    for path in csv_paths:
        condition, run = condition_and_run(path, root)
        rows = read_csv(path)
        intro = next((r for r in rows if r.get("row_type") == "document_intro"), {})
        codings = [r for r in rows if r.get("row_type") == "coding"]
        directions = next((r for r in rows if r.get("row_type") == "research_directions"), {})
        explanation = intro.get("code_count_explanation", "")
        direction_text = directions.get("research_directions", "")
        n_directions = len(re.findall(r"^方向 \d+", direction_text, flags=re.M))
        synthesis_failed = "自動研究方向整理未成功" in direction_text or "沒有足夠" in direction_text
        scores = [int(r["analytic_score"]) for r in codings if str(r.get("analytic_score", "")).strip().isdigit()]
        interviewer_main = sum(1 for r in codings if "訪員" in (r.get("speaker") or ""))
        transcript = intro.get("source_file") or (codings[0].get("source_file") if codings else path.stem)
        summary.append({
            "條件代號": condition, "run": run, "逐字稿": transcript, "CSV": str(path),
            "code 列數": len(codings),
            "設定下限": first_int(RE_MINIMUM, explanation),
            "可定位初始候選": first_int(RE_INITIAL, explanation),
            "通過門檻": first_int(RE_ELIGIBLE, explanation),
            "最後保留": first_int(RE_FINAL, explanation) or (str(len(codings)) if "共保留" in explanation else ""),
            "略過片段": first_int(RE_SKIPPED, explanation) or "0",
            "研究方向數": n_directions, "方向整理失敗": "Y" if synthesis_failed else "N",
            "平均 analytic_score": round(statistics.mean(scores), 1) if scores else "",
            "訪員主句數": interviewer_main,
            "code_count_explanation": explanation,
        })
        segsets.setdefault((condition, transcript), {})[run] = {r.get("segment_id", "") for r in codings}
        for index, r in enumerate(codings, 1):
            codes_long.append({
                "條件代號": condition, "run": run, "逐字稿": transcript, "序號": index,
                "segment_id": r.get("segment_id", ""), "speaker": r.get("speaker", ""),
                "evidence_quote": r.get("evidence_quote", ""), "code": r.get("code", ""),
                "why_this_code": r.get("why_this_code", "") or r.get("rationale", ""),
                "analytic_score": r.get("analytic_score", ""),
                "supporting_segment_ids": r.get("supporting_segment_ids", ""),
                "evidence_context": r.get("evidence_context", ""),
                "research_relevance": r.get("research_relevance", ""), "behavior_pattern": r.get("behavior_pattern", ""),
                "evidence_strength": r.get("evidence_strength", ""), "opportunity_potential": r.get("opportunity_potential", ""),
                "inference_risk": r.get("inference_risk", ""),
            })

    write_csv(out / "summary.csv", summary, list(summary[0].keys()))
    write_csv(out / "codes_long.csv", codes_long, list(codes_long[0].keys()) if codes_long else ["條件代號"])

    # stability: Jaccard between runs of the same condition × transcript
    stability = []
    for (condition, transcript), runs in sorted(segsets.items()):
        for a, b in combinations(sorted(runs), 2):
            union = runs[a] | runs[b]
            jac = len(runs[a] & runs[b]) / len(union) if union else ""
            stability.append({"條件代號": condition, "逐字稿": transcript, "run A": a, "run B": b,
                              "Jaccard(主句 segment_id)": round(jac, 3) if jac != "" else ""})
    write_csv(out / "stability.csv", stability, ["條件代號", "逐字稿", "run A", "run B", "Jaccard(主句 segment_id)"])

    # planted-pattern auto-check
    if args.key:
        key_rows = read_csv(Path(args.key))
        checks = []
        by_ct: dict[tuple[str, str, str], list[dict]] = {}
        for row in codes_long:
            by_ct.setdefault((row["條件代號"], row["run"], row["逐字稿"]), []).append(row)
        conditions_runs = sorted({(r["條件代號"], r["run"]) for r in codes_long})
        for key in key_rows:
            anchor = norm(key.get("關鍵句（逐字）") or key.get("關鍵句") or "")
            transcript = key.get("逐字稿", "")
            if not anchor:
                continue
            for condition, run in conditions_runs:
                rows = [r for (c, rr, t), lst in by_ct.items() if c == condition and rr == run and t == transcript for r in lst]
                hit = ""
                for r in rows:
                    hay = norm(r["evidence_quote"]) + "|" + norm(r["evidence_context"])
                    if anchor in hay:
                        hit = r
                        break
                checks.append({
                    "埋點ID": key.get("埋點ID", ""), "逐字稿": transcript, "類型": key.get("類型", ""),
                    "條件代號": condition, "run": run,
                    "自動比對(命中/未命中)": "命中" if hit else "未命中",
                    "人工得分(1/0.5/0)": "",
                    "對應 code（貼上）": hit["code"] if hit else "",
                    "命中序號": hit["序號"] if hit else "",
                })
        write_csv(out / "planted_autocheck.csv", checks,
                  ["埋點ID", "逐字稿", "類型", "條件代號", "run", "自動比對(命中/未命中)", "人工得分(1/0.5/0)", "對應 code（貼上）", "命中序號"])
        hits = sum(1 for c in checks if c["自動比對(命中/未命中)"] == "命中")
        print(f"埋點自動比對：{hits}/{len(checks)} 命中（僅供初稿，請人工確認並給分）")

    print(f"summary: {len(summary)} 份 CSV；codes_long: {len(codes_long)} 筆 code；stability: {len(stability)} 組 → {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
