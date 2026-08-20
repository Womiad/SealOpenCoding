#!/usr/bin/env python3
"""Local-LLM assisted open coding for interview transcripts.

The model never supplies the final quote. It returns segment IDs; the program
joins those IDs back to the verbatim source text before writing CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


SUPPORTED = {".txt", ".md", ".docx"}
SENTENCE_END = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z0-9\[（(])")
SPEAKER = re.compile(r"^\s*([^：:\n]{1,30})[：:]\s*(.*)$")
RESPONDENT_DASH = re.compile(r"^\s*[-–—]\s*(.+)$")
TURN_PLUS = re.compile(r"^\s*\+\s*(.+)$")
MINIMAL_RESPONSE = re.compile(
    r"^(?:嗯+|好+|對+|是+|否|不是|有|無|沒有|會|不會|可以|不可以|還好)[，,。.!！?？\s]*$"
)


def _is_speaker_label(value: str) -> bool:
    """Distinguish a compact speaker tag from an in-sentence colon/time."""
    label = value.strip()
    if not label or len(label) > 12:
        return False
    if re.search(r"[。！？!?；;，,./]", label):
        return False
    if re.search(r"\d", label):
        return bool(
            re.fullmatch(r"[A-Za-z]{1,3}\d{1,3}", label)
            or re.fullmatch(r"(?:受訪者|訪員|訪談者|研究者|參與者|主持人)\d{1,3}", label)
        )
    return bool(re.fullmatch(r"[\w\u3400-\u9fff（）() -]+", label))


@dataclass(frozen=True)
class Segment:
    id: str
    line: int
    speaker: str
    text: str
    context_before: str = ""
    context_after: str = ""

    @property
    def full_context(self) -> str:
        target_label = self.speaker or "未標記講者"
        parts = [self.context_before, f"【目標片段 {self.id}｜{target_label}】{self.text}", self.context_after]
        return "\n".join(part for part in parts if part)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8-sig", "utf-8", "cp950"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"無法判斷文字編碼：{path}")
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(ns + "p"):
            text = "".join(node.text or "" for node in paragraph.iter(ns + "t"))
            if text.strip():
                paragraphs.append(text)
        return "\n".join(paragraphs)
    raise ValueError(f"不支援的格式：{path.suffix}")


def segment_transcript(text: str, context_radius: int = 6) -> list[Segment]:
    """Split a transcript and attach a broad reading window to each segment.

    The window is only material the model may inspect.  Each coding later
    selects its own, usually much smaller, set of supporting segments.
    """
    segments: list[Segment] = []
    counter = 1
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        dash_match = RESPONDENT_DASH.match(line)
        plus_match = TURN_PLUS.match(line)
        match = SPEAKER.match(line)
        if plus_match:
            # A leading plus is also used as a turn marker in some exports.
            # It is deliberately preserved as a generic speaker marker: the
            # research role must come from the guide or the surrounding talk.
            speaker, content = "講者（+標記）", plus_match.group(1).strip()
        elif dash_match:
            # A leading dash distinguishes a speaker/turn in some transcript
            # formats, but does not identify that person's research role.
            speaker, content = "講者（-標記）", dash_match.group(1).strip()
        elif match and _is_speaker_label(match.group(1)):
            speaker, content = match.group(1).strip(), match.group(2).strip()
        else:
            speaker, content = "", line
        # Preserve each meaningful sentence while retaining its source line.
        sentences = [part.strip() for part in SENTENCE_END.split(content) if part.strip()]
        if not sentences:
            sentences = [content]
        for sentence in sentences:
            segments.append(Segment(f"S{counter:06d}", line_number, speaker, sentence))
            counter += 1
    def context_line(segment: Segment) -> str:
        label = segment.speaker or "未標記講者"
        return f"【{segment.id}｜{label}】{segment.text}"

    enriched: list[Segment] = []
    for index, segment in enumerate(segments):
        before = "\n".join(context_line(item) for item in segments[max(0, index - context_radius):index])
        after = "\n".join(context_line(item) for item in segments[index + 1:index + context_radius + 1])
        enriched.append(Segment(
            segment.id,
            segment.line,
            segment.speaker,
            segment.text,
            context_before=before,
            context_after=after,
        ))
    return enriched


def chunk_segments(segments: list[Segment], max_chars: int, max_segments: int = 30) -> Iterable[list[Segment]]:
    chunk: list[Segment] = []
    size = 0
    for segment in segments:
        item_size = len(segment.text) + len(segment.speaker) + 30
        if chunk and (size + item_size > max_chars or len(chunk) >= max_segments):
            yield chunk
            chunk, size = [], 0
        chunk.append(segment)
        size += item_size
    if chunk:
        yield chunk


def ollama_chat(host: str, model: str, system: str, prompt: str, timeout: int) -> dict:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"模型請求逾時（{timeout} 秒）") from exc
    except urllib.error.HTTPError as exc:
        if exc.code in {408, 504}:
            raise RuntimeError(f"模型請求逾時（HTTP {exc.code}，上限 {timeout} 秒）") from exc
        raise RuntimeError(f"Ollama HTTP 錯誤（{exc.code}）：{exc.reason}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise RuntimeError(f"模型請求逾時（{timeout} 秒）") from exc
        raise RuntimeError(f"無法連線 Ollama（{url}）：{exc}") from exc
    content = body.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模型沒有回傳有效 JSON：{content[:500]}") from exc


def build_prompt(
    guide: str,
    chunk: list[Segment],
    respondent_only: bool = False,
    source_file: str = "",
    participant_role: str = "",
) -> tuple[str, str]:
    system = """你是嚴謹的質性研究者，執行初始 open coding（initial coding）。
研究指引只是 sensitizing framework，用來說明研究情境與注意方向，不是封閉 codebook。
code 必須從受訪者實際說法歸納，不得把指引中的詞機械套入，也不得為符合研究目的而創造原文沒有的意圖、因果或介入效果。

逐一掃描每個 segment。優先 coding 受訪者的經驗、行動、感受、評價、需求、顧慮、選擇與因應方式。
若 speaker 是「訪員／研究者」，不要 coding。若 speaker 空白且內容主要是提問、功能介紹、流程說明或引導語，也不要 coding。
逐字稿中的「講者（+標記）」與「講者（-標記）」只表示原文以符號標示一次發言，不代表受訪者、訪員、病人、陪伴者或家屬。
只有 speaker 欄明確寫出角色時才能依角色判斷；角色未明時，依發言內容區分經驗陳述與訪談提問，不得從破折號推測角色。
缺少理由、條件或具體經驗的簡短同意不可單獨 coding；不得用訪員問題替短答補出需求或態度。
code 必須同時受目標原文與 full_context 支持；不得把訪員的假設當作講者已經歷的事實。
若主要參與者是在代述他人的經驗，code 必須保留「觀察到／認為／推測」等證據身分，不得改寫成被描述者的第一人稱選擇、需求或內在動機。

code 要寫成一個可獨立理解的邏輯分析句，通常包含「情境／前因＋講者的想法或行為＋結果／意義」，約 20–60 個中文字。
建議句型：「在［原文支持的情境］下，［講者角色］［想法或行為］，呈現［原文支持的結果或意義］。」
若原文明確說出因果，應在 code 中用「因為／所以／導致／使得」呈現完整前因後果。
若上下文沒有明確因果，改用「在……情境下，講者……，呈現……」描述可觀察的關係，絕不可為了句型補造原因或結果。
可保留貼近原話的 in-vivo 用語，但不能只輸出缺少情境與行動關係的短 topic label。
每個有分析價值的 segment 通常給 1–3 個互不重複的 code；無分析價值則不標。
不要漏掉受訪者主動敘述的長回答、強烈感受、拒絕、顧慮、痛苦經驗或具體因應行動；如實 coding，但不得做臨床診斷。
候選 code 至少應呈現一項分析價值：獨特觀點、明確選擇條件、反覆行為、因應策略、矛盾、界線、轉變、強烈評價或意外反應。
不要為人口資料、一般作息、單純時間地點、理所當然的事實或只是把句子換成名詞的主題標籤產生 code，除非原文清楚顯示它和研究現象形成具體關係。
每筆候選另用 0–4 評分，不得因句子較長或出現「因為／所以」就加分：
- research_relevance：0=與研究問題無關；1=只有遙遠背景；2=提供研究現象的重要脈絡；3=直接回應研究現象；4=揭示核心張力、條件或機制。
- behavior_pattern：0=單一事實；1=一次性行動；2=具體策略或情境反應；3=反覆／條件式模式；4=包含觸發、行動與後果的清楚機制。
- evidence_strength：0=訪員話語或無支持；1=主要靠問題／短答推論；2=原句部分支持；3=原句直接支持；4=多句脈絡一致且明確。
- opportunity_potential：0=無；1=一般背景；2=顯示可追問的需求／阻礙；3=明確設計或研究機會；4=具體情境、對象、條件與界線皆清楚。
- inference_risk：0=完全貼近原話；1=輕微概念化；2=合理但需複核；3=把他人觀察當內在動機或延伸較多；4=訪員移植／角色錯置／補造因果。
每一個 code 都必須有自己專屬的 rationale，說明原句中的哪個意思支持這個 code，以及兩者如何相關；
不要只重複 code 名稱，不要在 rationale 加入原文未表達的工具需求、效果或研究價值。
若同一句有多個 code，各筆 rationale 必須分別解釋各自的判斷。
你只能使用輸入中存在的 segment_id，不得自行創造、合併或改寫句子。
每筆 code 必須另外提供 supporting_segment_ids，彈性指出真正支持該 code 的原文範圍：
- 只選必要的 segments，不要因為看得到前後文就全部選入；重點只在一句時只列該句。
- 若判斷依賴「功能說明／情境 → 實際體驗 → 反應 → 評價」等長事件，可依原始順序列出 5 句以上；範圍不能超過輸入提供的前後文搜尋邊界。
- supporting segments 可包含用來交代情境的訪員話語，但主要 segment_id 必須是受訪者自己的實質想法、行為或評價；不可只用訪員說明產生 code。
- 選入的每一句都必須對理解該 code 有實質作用，刪掉不影響判斷的句子就不要選。
每一筆都必須提供 evidence_quote：從該 segment 原封不動複製、足以支持該 code 的連續短句，
不可改寫、不可把不同 segment 的文字拼在一起。segment_id 必須是 evidence_quote 所在的那一段。
只輸出 JSON 物件，格式為：
{"codings":[{"segment_id":"S000001","supporting_segment_ids":["S000001"],"evidence_quote":"原文中的精確短句","code":"包含情境與前因後果的完整邏輯 code","rationale":"上下文如何支持此判斷","research_relevance":0,"behavior_pattern":0,"evidence_strength":0,"opportunity_potential":0,"inference_risk":0,"score_reason":"為何與研究相關且推論風險可接受","confidence":0.0}]}
confidence 為 0 到 1。同一 segment 的每個 code 各自一筆。沒有適合內容時回傳 {"codings":[]}。"""
    items = [
        {
            "segment_id": s.id,
            "speaker": s.speaker,
            "coding_allowed": not respondent_only or s.speaker == "受訪者",
            "text": s.text,
        }
        for s in chunk
    ]
    boundary_before = chunk[0].context_before if chunk else ""
    boundary_after = chunk[-1].context_after if chunk else ""
    prompt = (
        f"來源檔名：{source_file or '（未提供）'}\n"
        f"主要受訪者角色：{participant_role or '未由檔名判定'}\n"
        "若研究指引定義了檔名與受訪者角色的對應，可據此辨識本文件的主要受訪者；"
        "但檔名不代表每一段都是主要受訪者發言，仍須排除訪員與其他講者。\n\n"
        "研究背景、研究問題與希望關注的概念如下：\n"
        f"---\n{guide}\n---\n\n"
        "以下是一個按原始順序排列的連續對話視窗。每個 segment 只出現一次；"
        "可用相鄰 segment 理解上下文，但 evidence_quote 必須逐字取自被 coding 的目標 segment。"
        "補充上下文不能當主要 coding 目標，但其中的 ID 可在確有必要時列入 supporting_segment_ids。\n"
        f"視窗開始前的補充上下文：\n{boundary_before or '（無）'}\n\n"
        "可 coding 的視窗 segments：\n"
        + json.dumps(items, ensure_ascii=False)
        + f"\n\n視窗結束後的補充上下文：\n{boundary_after or '（無）'}"
    )
    return system, prompt


def _rubric_value(row: dict, name: str, default: int) -> int:
    try:
        return max(0, min(4, int(round(float(row.get(name, default))))))
    except (TypeError, ValueError):
        return default


def quality_dimensions(row: dict, segment: Segment) -> dict[str, int | str]:
    """Normalize model rubric fields; legacy results receive neutral defaults."""
    text_length = len(re.sub(r"[^\w\u3400-\u9fff]", "", segment.text))
    evidence_default = 3 if text_length >= 20 else 2
    values = {
        "research_relevance": _rubric_value(row, "research_relevance", 2),
        "behavior_pattern": _rubric_value(row, "behavior_pattern", 2),
        "evidence_strength": _rubric_value(row, "evidence_strength", evidence_default),
        "opportunity_potential": _rubric_value(row, "opportunity_potential", 1),
        "inference_risk": _rubric_value(row, "inference_risk", 1),
    }
    score = (
        values["research_relevance"] * 10
        + values["behavior_pattern"] * 7
        + values["evidence_strength"] * 6
        + values["opportunity_potential"] * 2
        - values["inference_risk"] * 8
    )
    values["analytic_score"] = max(0, min(100, score))
    values["score_reason"] = str(row.get("score_reason", "")).strip()
    return values


def quality_eligible(row: dict) -> bool:
    """Hard gate: relevance and evidence precede ranking or minimum backfill."""
    return (
        int(row.get("research_relevance", 2)) >= 2
        and int(row.get("evidence_strength", 2)) >= 2
        and int(row.get("inference_risk", 1)) <= 2
    )


def code_count_explanation(
    minimum: int,
    initial_count: int,
    eligible_count: int,
    final_count: int,
    skipped_segments: int,
    focused_refinement: bool,
) -> str:
    """Explain code quantity using observed pipeline counts, without LLM guesses."""
    if not focused_refinement:
        message = (
            f"本次未啟用「精選重要 code」，最低數量設定不生效；"
            f"共保留 {final_count} 個可定位的初始 code。"
        )
    elif minimum == 0:
        message = (
            f"本次未設定最低 code 數；海豹從 {initial_count} 個可定位初始候選中，"
            f"辨識出 {eligible_count} 個通過品質門檻，最後保留 {final_count} 個。"
        )
        if final_count == 0:
            message += "沒有合格候選時允許輸出 0 個，以免用無關或高推論風險內容湊數。"
    elif final_count < minimum and eligible_count < minimum:
        message = (
            f"設定至少 {minimum} 個，但 {initial_count} 個可定位初始候選中只有 "
            f"{eligible_count} 個通過研究相關性、語證強度與推論風險門檻，"
            f"最後保留 {final_count} 個。海豹不使用未通過門檻的 code 補足數量。"
        )
    elif final_count < minimum:
        message = (
            f"設定至少 {minimum} 個，且有 {eligible_count} 個候選通過品質門檻，"
            f"但聚焦精選只成功保留 {final_count} 個；可能是精選回傳無效、重複或無法再次核對語證。"
            "建議查看處理 log 並人工複核合格候選。"
        )
    else:
        message = (
            f"設定至少 {minimum} 個；共有 {initial_count} 個可定位初始候選、"
            f"{eligible_count} 個通過品質門檻，聚焦後保留 {final_count} 個。"
        )
    if initial_count > eligible_count:
        message += f" 另有 {initial_count - eligible_count} 個候選未通過品質門檻。"
    if skipped_segments:
        message += f" 此外有 {skipped_segments} 個片段因持續逾時或無法可靠定位而略過，建議人工抽查原文。"
    return message


def recompute_quality_score(row: dict) -> None:
    row["analytic_score"] = max(0, min(100,
        int(row.get("research_relevance", 2)) * 10
        + int(row.get("behavior_pattern", 2)) * 7
        + int(row.get("evidence_strength", 2)) * 6
        + int(row.get("opportunity_potential", 1)) * 2
        - int(row.get("inference_risk", 1)) * 8
    ))


def looks_like_interviewer_turn(segment: Segment) -> bool:
    text = segment.text.strip()
    if segment.speaker in {"訪員", "訪談者", "研究者", "主持人"}:
        return True
    generic_marker = segment.speaker in {"講者（+標記）", "講者（-標記）"}
    if segment.speaker and not generic_marker:
        return False
    if re.search(r"(?:我們|本研究|這個).{0,16}(?:系統|工具|功能|產品).{0,12}(?:會|可以|提供)", text):
        return True
    if ("?" in text or "？" in text) and re.search(r"(?:你|您|會不會|有沒有|覺得|願意|想不想)", text):
        return True
    return False


CONTEXT_SEGMENT = re.compile(r"^【(?:目標片段 )?(S\d{6})｜[^】]*】(.*)$")


def selected_evidence_context(segment: Segment, requested_ids: object) -> tuple[list[str], str]:
    """Ground a model-selected evidence range in the exact visible source text."""
    available: dict[str, str] = {}
    order: list[str] = []
    for line in segment.full_context.splitlines():
        match = CONTEXT_SEGMENT.match(line)
        if not match:
            continue
        segment_id = match.group(1)
        available[segment_id] = line
        order.append(segment_id)

    requested = [str(value) for value in requested_ids] if isinstance(requested_ids, list) else []
    requested.append(segment.id)
    wanted = set(requested)
    selected_ids = [segment_id for segment_id in order if segment_id in wanted]
    if segment.id not in selected_ids:
        # The primary source segment must always remain visible even if a
        # malformed model response omitted it.
        selected_ids = [value for value in selected_ids if value != segment.id] + [segment.id]
        selected_ids.sort(key=order.index)
    evidence_context = "\n".join(available[value] for value in selected_ids)
    return selected_ids, evidence_context


def _context_ids(text: str) -> list[str]:
    """Return source IDs visible in a stored context window, in source order."""
    ids: list[str] = []
    for line in str(text).splitlines():
        match = CONTEXT_SEGMENT.match(line)
        if match and match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def _evidence_text_from_ids(
    requested_ids: object,
    primary_id: str,
    visible_ids: set[str],
    segments: list[Segment],
) -> tuple[list[str], str]:
    """Rebuild selected context from document source, never model-authored text."""
    requested = [str(value) for value in requested_ids] if isinstance(requested_ids, list) else []
    wanted = {value for value in requested if value in visible_ids}
    wanted.add(primary_id)
    selected = [segment for segment in segments if segment.id in wanted]
    ids = [segment.id for segment in selected]
    lines = [f"【{segment.id}｜{segment.speaker or '未標記講者'}】{segment.text}" for segment in selected]
    return ids, "\n".join(lines)


def refine_evidence_ranges(
    guide: str,
    codings: list[dict],
    segments: list[Segment],
    args: argparse.Namespace,
) -> list[dict]:
    """Correct each code's primary quote and flexibly select necessary context.

    Initial and focused coding optimize code quality. This separate, optional
    pass asks the model only where the supporting episode begins and ends. All
    returned IDs are validated and joined back to the original transcript.
    A failed batch preserves its existing rows so overnight jobs keep going.
    """
    if not codings:
        return codings
    by_segment = {segment.id: segment for segment in segments}
    batch_size = 5
    batches = [codings[index:index + batch_size] for index in range(0, len(codings), batch_size)]
    corrected: list[dict] = []

    system = """你是質性研究的語證校對者。code 已經決定；你的工作只是在逐字稿可見範圍內，選出真正支持每個 code 的彈性上下文。

每筆都要重新判斷：
1. primary_segment_id 必須是參與者自己的實質經驗、想法、行為或評價所在句；訪員的問題、功能介紹、重述或總結只能當 supporting context，不能當主句。若 current primary 是問「你／您」的問句，絕不可原樣選回，必須在可見範圍尋找後續實質回答。
2. 先把 code 拆成「情境／前因、想法或行動、理由、結果、比較或界線」等實際存在的部分，再逐項確認是哪個原句支持。code 是先前綜合上下文寫成的分析句，不能因為 code 本身很完整，就假定 current primary 一句已經支持全部內容。
3. supporting_segment_ids 只保留理解 code 必要的句子，並包含 primary。只有同一句明確包含 code 的所有實質部分時才能只選一句；若不同句分別提供情境、操作、反應、理由、比較或評價，必須全部選入。「功能說明／實際體驗 → 反應 → 理由／比較 → 評價」的長事件通常需要 5 句以上。
4. 參與者使用「這個、那個、這樣、會、不會、對」等依賴前文的回答時，必須納入讓指涉可理解的問題或功能情境；比較兩種經驗時，必須納入被比較的兩側證據。
5. 不要機械地取前後固定句數。每一句都要與 code 的情境、行動、理由或結果直接相關；移除後不影響理解的寒暄、換題或枝節不要選。
6. 「講者（+標記）」與「講者（-標記）」都只是原稿的發言標記，不自動代表任何研究角色。依發言內容、研究指引及對話關係判斷。
7. 只能使用該 candidate 的 visible_segment_ids，不得創造 ID，也不得改寫 code 或原文。supporting_segments 要為每個選入 ID 標明它提供的語證功能，不能填籠統的「相關」。

只輸出 JSON：
{"contexts":[{"candidate_id":"C00001","primary_segment_id":"S000001","supporting_segment_ids":["S000001"],"supporting_segments":[{"segment_id":"S000001","function":"想法或行動"}],"reason":"逐項說明 code 的哪些部分由哪些句子支持，以及為何邊界到此為止"}]}"""

    for batch_index, batch in enumerate(batches, 1):
        candidates: list[dict] = []
        candidate_rows: dict[str, dict] = {}
        visible_by_candidate: dict[str, set[str]] = {}
        for local_index, coding in enumerate(batch, 1):
            candidate_id = f"C{local_index:05d}"
            visible = _context_ids(coding.get("full_context", ""))
            if coding.get("segment_id") not in visible:
                visible.append(str(coding.get("segment_id", "")))
            visible = [value for value in visible if value in by_segment]
            candidate_rows[candidate_id] = coding
            visible_by_candidate[candidate_id] = set(visible)
            candidates.append({
                "candidate_id": candidate_id,
                "current_primary_segment_id": coding.get("segment_id", ""),
                "current_supporting_segment_ids": coding.get("supporting_segment_ids", []),
                "code": coding.get("code", ""),
                "why_this_code": coding.get("rationale", ""),
                "visible_segment_ids": visible,
                "visible_context": coding.get("full_context", ""),
            })

        print(f"  語證範圍校正 {batch_index}/{len(batches)}：{len(batch)} 個 code", flush=True)
        prompt = (
            "研究指引（只用來辨識研究角色與研究情境，不可擴張原文）：\n"
            f"---\n{guide}\n---\n\n待校正資料：\n"
            + json.dumps(candidates, ensure_ascii=False)
        )
        try:
            result = ollama_chat(args.host, args.model, system, prompt, args.timeout)
            contexts = result.get("contexts", [])
            if not isinstance(contexts, list):
                raise RuntimeError("語證校正結果的 contexts 必須是陣列")
        except RuntimeError as exc:
            print(f"    警告：語證範圍校正失敗，保留原範圍並繼續：{exc}", flush=True)
            corrected.extend(batch)
            continue

        proposals = {
            str(item.get("candidate_id", "")): item
            for item in contexts if isinstance(item, dict)
        }
        for candidate_id, original in candidate_rows.items():
            proposal = proposals.get(candidate_id)
            if proposal is None:
                corrected.append(original)
                continue
            primary_id = str(proposal.get("primary_segment_id", ""))
            visible_ids = visible_by_candidate[candidate_id]
            primary = by_segment.get(primary_id)
            normalized = re.sub(r"^[+\-–—\s]+", "", primary.text.strip()) if primary else ""
            meaningful = re.sub(r"[^\w\u3400-\u9fff]", "", normalized)
            if (
                primary is None
                or primary_id not in visible_ids
                or len(meaningful) < 6
                or MINIMAL_RESPONSE.fullmatch(normalized)
                or looks_like_interviewer_turn(primary)
            ):
                corrected.append(original)
                continue
            requested_support = proposal.get("supporting_segment_ids", [])
            if not isinstance(requested_support, list):
                requested_support = []
            detailed_support = proposal.get("supporting_segments", [])
            if isinstance(detailed_support, list):
                requested_support.extend(
                    item.get("segment_id") for item in detailed_support if isinstance(item, dict)
                )
            supporting_ids, evidence_context = _evidence_text_from_ids(
                requested_support,
                primary_id,
                visible_ids,
                segments,
            )
            row = dict(original)
            row.update({
                "segment_id": primary_id,
                "supporting_segment_ids": supporting_ids,
                "evidence_context": evidence_context,
                "evidence_quote": primary.text,
                "context_before": primary.context_before,
                "context_after": primary.context_after,
                "full_context": primary.full_context,
            })
            corrected.append(row)
    return corrected


def validate_codings(
    result: dict,
    segments: list[Segment],
    allowed_ids: set[str] | None = None,
) -> list[dict]:
    rows = result.get("codings", [])
    if not isinstance(rows, list):
        raise RuntimeError("模型 JSON 的 codings 必須是陣列")
    by_id = {segment.id: segment for segment in segments}
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code", "")).strip()
        evidence = str(row.get("evidence_quote", "")).strip()
        if not code or not evidence:
            continue
        requested_id = str(row.get("segment_id", ""))
        requested = by_id.get(requested_id)
        if requested is not None and evidence in requested.text:
            grounded = requested
        else:
            # Repair a wrong model-supplied ID only when its verbatim evidence
            # unambiguously identifies another source segment in this chunk.
            matches = [segment for segment in segments if evidence in segment.text]
            if len(matches) != 1:
                continue
            grounded = matches[0]
        if allowed_ids is not None and grounded.id not in allowed_ids:
            continue
        normalized_text = re.sub(r"^[+\-–—\s]+", "", grounded.text.strip())
        if MINIMAL_RESPONSE.fullmatch(normalized_text):
            continue
        meaningful_chars = re.sub(r"[^\w\u3400-\u9fff]", "", grounded.text)
        if len(meaningful_chars) < 6 or looks_like_interviewer_turn(grounded):
            continue
        if len(re.sub(r"[^\w\u3400-\u9fff]", "", evidence)) <= 12 and len(meaningful_chars) >= 20:
            evidence = grounded.text
        try:
            confidence = min(1.0, max(0.0, float(row.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str(row.get("rationale", "")).strip()
        dimensions = quality_dimensions(row, grounded)
        supporting_ids, evidence_context = selected_evidence_context(
            grounded, row.get("supporting_segment_ids", [])
        )
        cleaned.append({
            "segment_id": grounded.id,
            "supporting_segment_ids": supporting_ids,
            "evidence_context": evidence_context,
            "evidence_quote": evidence,
            "code": code,
            "rationale": rationale,
            "confidence": confidence,
            **dimensions,
            "context_before": grounded.context_before,
            "context_after": grounded.context_after,
            "full_context": grounded.full_context,
        })
    return cleaned


def output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{source.stem}_open_coding.csv"


def participant_role_from_filename(source: Path, guide: str) -> str:
    """Read a generic filename-prefix role rule from the private guide."""
    if not source.stem:
        return ""
    prefix = re.escape(source.stem[0])
    patterns = (
        rf"(?:檔名[^\n]{{0,20}})?{prefix}\s*(?:開頭)?\s*(?:代表|表示|是|為)\s*([^\n，,。；;]{{1,30}})",
        rf"(?:^|\n)\s*{prefix}\s*[:：=]\s*([^\n，,。；;]{{1,30}})",
    )
    for pattern in patterns:
        match = re.search(pattern, guide, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def batch_output_paths(sources: list[Path], output_dir: Path) -> dict[Path, Path]:
    """Give every batch source a stable destination, including duplicate names."""
    stem_counts: dict[str, int] = {}
    for source in sources:
        key = source.stem.casefold()
        stem_counts[key] = stem_counts.get(key, 0) + 1
    destinations = {}
    for source in sources:
        resolved = source.resolve()
        if stem_counts[source.stem.casefold()] == 1:
            destinations[resolved] = output_path(source, output_dir)
        else:
            fingerprint = hashlib.sha1(str(resolved).casefold().encode("utf-8")).hexdigest()[:8]
            destinations[resolved] = output_dir / f"{source.stem}_{fingerprint}_open_coding.csv"
    return destinations


def _shorten(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 1) // 2
    return text[:half] + "…" + text[-half:]


def _focused_select(
    guide: str,
    codings: list[dict],
    args: argparse.Namespace,
    maximum: int,
    stage: str,
    minimum: int = 0,
) -> list[dict] | None:
    """Run one evidence-grounded focused-selection request."""
    candidates = []
    by_id: dict[str, dict] = {}
    for index, coding in enumerate(codings, 1):
        candidate_id = f"C{index:05d}"
        by_id[candidate_id] = coding
        candidates.append({
            "candidate_id": candidate_id,
            "segment_id": coding["segment_id"],
            "evidence": _shorten(coding.get("evidence_quote", "")),
            "selected_context": _shorten(coding.get("evidence_context", ""), 520),
            "full_context": _shorten(coding.get("full_context", ""), 520),
            "initial_code": coding["code"],
            "initial_reason": _shorten(coding.get("rationale", ""), 140),
            "quality_rubric": {
                "research_relevance": coding.get("research_relevance", 2),
                "behavior_pattern": coding.get("behavior_pattern", 2),
                "evidence_strength": coding.get("evidence_strength", 2),
                "opportunity_potential": coding.get("opportunity_potential", 1),
                "inference_risk": coding.get("inference_risk", 1),
                "analytic_score": coding.get("analytic_score", 0),
                "score_reason": coding.get("score_reason", ""),
            },
        })

    system = """你是資深質性研究者，執行 initial coding 之後的 focused refinement。
你要比較整份訪談的候選 codes，少量保留真正有分析價值者，而不是摘要主題或追求覆蓋率。

先做品質門檻，再比較分數。只有 research_relevance >= 2、evidence_strength >= 2、inference_risk <= 2 才能保留。
優先保留直接回應研究現象、呈現具體／反覆／條件式行為模式、因應策略、矛盾、轉變、界線、拒絕、顧慮與接受條件者。
「研究機會」是從明確需求、阻礙、未被滿足的目標或設計條件產生的可追問方向；不可把一般生活故事硬連成產品機會。

刪除：人口資料、一般作息、單純事實、理所當然的描述、訪員觀點、短答延伸、
只把句子換成名詞的 topic code、和研究問題無關的細節，以及彼此重複但較弱的證據。
不要因研究涉及某種工具或介入就把一般背景內容解釋成工具需求。

必須把保留的 initial_code 改寫成可獨立理解的邏輯分析句，包含情境／前因、講者的想法或行為，以及結果／意義。
使用「在［情境］下，［角色］［想法或行為］，呈現［結果／意義］」等可獨立理解的分析句，
而非缺少情境、行動與關係的短標籤。
只有 full_context 明確支持因果時才能寫「因為／所以」；沒有明確因果時，用「在……情境下，講者……，呈現……」描述關係，不得補造因果。
若多筆共同呈現一個模式，可給它們相同或相互呼應的 refined_code；只保留最有力的 1–3 筆證據。
每筆 selection 必須包含 evidence_quote：從該 candidate 的 evidence 原封不動複製一段連續文字，不能改寫或跨筆拼接。
重新核對每個保留項目的五個 0–4 rubric 分數；分數定義與候選中的 quality_rubric 相同，不得因 code 較長或含因果連接詞就提高。
不得創造 candidate_id。只輸出 JSON：
{"selections":[{"candidate_id":"C00001","evidence_quote":"候選中的精確原文","refined_code":"精煉 code","rationale":"原文如何支持此特定觀點或行為模式","research_relevance":0,"behavior_pattern":0,"evidence_strength":0,"opportunity_potential":0,"inference_risk":0,"score_reason":"評分理由"}]}"""
    quantity_instruction = (
        f"請保留至少 {minimum} 個、最多 {maximum} 個。\n"
        if minimum
        else f"請最多保留 {maximum} 個；沒有足夠分析價值時可以更少。\n"
    )
    prompt = (
        "研究指引如下（它是 sensitizing framework，不是必須套用的 code 清單）：\n"
        f"---\n{guide}\n---\n\n"
        f"這是{stage}，共有 {len(candidates)} 個候選。"
        + quantity_instruction
        + "候選資料：\n"
        + json.dumps(candidates, ensure_ascii=False)
    )

    last_error: RuntimeError | None = None
    # Focused selection should never stall the whole document. One model call
    # is enough; evidence-scored fallback handles malformed output instantly.
    for attempt in range(1):
        try:
            result = ollama_chat(args.host, args.model, system, prompt, args.timeout)
            selections = result.get("selections", [])
            if not isinstance(selections, list):
                raise RuntimeError("聚焦精選結果的 selections 必須是陣列")
            refined: list[dict] = []
            seen: set[tuple[str, str]] = set()
            for selection in selections:
                if not isinstance(selection, dict):
                    continue
                candidate_id = str(selection.get("candidate_id", ""))
                original = by_id.get(candidate_id)
                evidence = str(selection.get("evidence_quote", "")).strip()
                code = str(selection.get("refined_code", "")).strip()
                rationale = str(selection.get("rationale", "")).strip()
                if not evidence or not code or not rationale:
                    continue
                if original is None or evidence not in original.get("evidence_quote", ""):
                    matches = [
                        (other_id, other)
                        for other_id, other in by_id.items()
                        if evidence in other.get("evidence_quote", "")
                    ]
                    if len(matches) != 1:
                        continue
                    candidate_id, original = matches[0]
                if len(re.sub(r"[^\w\u3400-\u9fff]", "", evidence)) < 6:
                    continue
                key = (candidate_id, code)
                if key in seen:
                    continue
                seen.add(key)
                row = dict(original)
                row["code"] = code
                row["rationale"] = rationale
                for field in ("research_relevance", "behavior_pattern", "evidence_strength", "opportunity_potential", "inference_risk"):
                    if field in selection:
                        row[field] = _rubric_value(selection, field, int(row.get(field, 2)))
                row["score_reason"] = str(selection.get("score_reason", row.get("score_reason", ""))).strip()
                recompute_quality_score(row)
                if quality_eligible(row):
                    refined.append(row)
                if len(refined) >= maximum:
                    break
            if selections and not refined:
                raise RuntimeError("聚焦精選結果沒有有效的 candidate_id")
            return refined
        except RuntimeError as exc:
            last_error = exc
            if "無法連線 Ollama" in str(exc):
                raise
    print(f"    警告：{stage}失敗：{last_error}", flush=True)
    return None


def _focused_select_resilient(
    guide: str,
    codings: list[dict],
    args: argparse.Namespace,
    maximum: int,
    stage: str,
    minimum: int = 0,
) -> list[dict]:
    selected = _focused_select(guide, codings, args, maximum, stage, minimum)
    if selected is not None:
        return _ensure_minimum(selected, codings, minimum)[:maximum]
    eligible = [row for row in codings if quality_eligible(row)]
    fallback_count = min(len(eligible), max(minimum, maximum))
    fallback = sorted(eligible, key=lambda row: int(row.get("analytic_score", 0)), reverse=True)[:fallback_count]
    print(f"    {stage}改用 code 分數，選出最高分的 {len(fallback)} 個後繼續下一階段。", flush=True)
    return fallback


def _ensure_minimum(selected: list[dict], pool: list[dict], minimum: int) -> list[dict]:
    """Backfill only from candidates passing relevance/evidence/risk gates."""
    selected = [row for row in selected if quality_eligible(row)]
    if len(selected) >= minimum:
        return selected
    keys = {(row.get("segment_id"), row.get("evidence_quote")) for row in selected}
    result = list(selected)
    ranked_pool = sorted(
        (row for row in pool if quality_eligible(row)),
        key=lambda row: int(row.get("analytic_score", 0)), reverse=True,
    )
    for row in ranked_pool:
        key = (row.get("segment_id"), row.get("evidence_quote"))
        if key in keys:
            continue
        result.append(row)
        keys.add(key)
        if len(result) >= minimum:
            break
    if len(result) < minimum:
        print(f"    品質門檻後只有 {len(result)} 個，不以低相關或高推論風險 code 湊足最低 {minimum} 個。", flush=True)
    return result


def refine_codings(guide: str, codings: list[dict], args: argparse.Namespace) -> list[dict]:
    """Hierarchically select meaningful codes across one interview."""
    if len(codings) <= 1:
        return codings
    overall_minimum = min(len(codings), max(0, int(getattr(args, "min_codes", 10))))
    natural_maximum = max(8, min(40, round(len(codings) * 0.35)))
    overall_maximum = min(len(codings), max(natural_maximum, overall_minimum))
    if len(codings) <= 24:
        selected = _focused_select_resilient(
            guide, codings, args, overall_maximum, "最終聚焦精選", overall_minimum
        )
        return selected

    shortlist: list[dict] = []
    batches = [codings[index:index + 20] for index in range(0, len(codings), 20)]
    per_batch_floor = (overall_minimum + len(batches) - 1) // len(batches)
    for index, batch in enumerate(batches, 1):
        batch_maximum = min(len(batch), max(3, min(8, round(len(batch) * 0.40)), per_batch_floor))
        print(f"    初選 {index}/{len(batches)}：{len(batch)} 個候選", flush=True)
        shortlist.extend(_focused_select_resilient(
            guide, batch, args, batch_maximum, f"初選 {index}"
        ))
    if not shortlist:
        eligible = [row for row in codings if quality_eligible(row)]
        fallback_count = min(len(eligible), max(overall_minimum, overall_maximum))
        fallback = sorted(eligible, key=lambda row: int(row.get("analytic_score", 0)), reverse=True)[:fallback_count]
        print(f"    所有初選皆無保留項目，改用整篇 code 分數選出最高分的 {len(fallback)} 個。", flush=True)
        return fallback
    print(f"    跨批比較：從 {len(shortlist)} 個初選候選中辨識整體模式", flush=True)
    selected = _focused_select_resilient(
        guide,
        shortlist,
        args,
        min(overall_maximum, len(shortlist)),
        "最終跨訪談精選",
        min(overall_minimum, len(shortlist)),
    )
    return _ensure_minimum(selected, codings, overall_minimum)


def generate_document_synthesis(
    guide: str,
    source: Path,
    segments: list[Segment],
    codings: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Create a grounded reader cover page and tentative research directions."""
    transcript_items = []
    transcript_chars = 0
    for segment in segments:
        item = {"segment_id": segment.id, "speaker": segment.speaker or "未標記講者", "text": segment.text}
        size = len(segment.text) + len(segment.speaker) + 30
        if transcript_items and transcript_chars + size > 12000:
            break
        transcript_items.append(item)
        transcript_chars += size

    coding_items = []
    coding_by_id: dict[str, dict] = {}
    for index, coding in enumerate(codings, 1):
        candidate_id = f"C{index:05d}"
        coding_by_id[candidate_id] = coding
        coding_items.append({
            "candidate_id": candidate_id,
            "segment_id": coding.get("segment_id", ""),
            "evidence_quote": coding.get("evidence_quote", ""),
            "evidence_context": coding.get("evidence_context", ""),
            "code": coding.get("code", ""),
            "why_this_code": coding.get("rationale", ""),
            "analytic_score": coding.get("analytic_score", 0),
            "research_relevance": coding.get("research_relevance", 0),
            "behavior_pattern": coding.get("behavior_pattern", 0),
            "evidence_strength": coding.get("evidence_strength", 0),
            "opportunity_potential": coding.get("opportunity_potential", 0),
            "inference_risk": coding.get("inference_risk", 0),
        })

    system = """你是嚴謹的質性研究助理。請替一份訪談建立簡短文本簡介，並根據已通過原文定位的 codes 提出可能研究方向。
文本簡介只能寫逐字稿明確支持的資料；年齡、身分或背景未明時必須寫「未明」，不得由語氣猜測。特質是訪談中可觀察的偏好、態度或行為傾向，不可做人格或臨床診斷。
研究方向只是供研究者複核的暫定分析線索，不是研究結論。研究機會必須由受訪者明確需求、阻礙、未滿足目標、矛盾或接受條件導出，不可把一般故事直接變成產品功能。
每個方向必須引用至少一個輸入中存在的 candidate_id，並提供 shared_concept：它必須是該 code 與方向／可能發現／研究機會中都逐字出現的關鍵概念（至少兩個字），用來防止引用錯配。不得創造 ID。
只輸出 JSON：
{"profile":{"participant":"受訪者是誰／角色，未明則寫未明","age":"明確年齡或未明","identity_context":"其他明確身分或生活脈絡，未明則寫未明","characteristics":["有原文根據的特質"],"interview_summary":"本篇訪談內容簡介"},"research_directions":[{"direction":"研究方向名稱","possible_finding":"可能的質性發現","research_opportunity":"可供後續研究或設計追問的機會，不是功能結論","shared_concept":"code 與本方向逐字共有的概念","candidate_ids":["C00001"],"caution":"解讀限制或仍需確認處"}]}"""
    prompt = (
        f"來源文件：{source.name}\n\n研究指引：\n---\n{guide}\n---\n\n"
        "逐字稿片段（依原始順序；若因長度截斷，不得推論未提供部分）：\n"
        + json.dumps(transcript_items, ensure_ascii=False)
        + "\n\n已通過定位與精選的 coding candidates：\n"
        + json.dumps(coding_items, ensure_ascii=False)
    )
    fallback_profile = {
        "participant": "未明", "age": "未明", "identity_context": "未明", "characteristics": [],
        "interview_summary": f"{source.name} 的訪談逐字稿；自動簡介產生失敗，請研究者直接複核原文。",
    }
    try:
        result = ollama_chat(args.host, args.model, system, prompt, args.timeout)
        raw_profile = result.get("profile", {})
        if not isinstance(raw_profile, dict):
            raw_profile = {}
        raw_characteristics = raw_profile.get("characteristics", [])
        characteristics = ([str(item).strip() for item in raw_characteristics if str(item).strip()]
                           if isinstance(raw_characteristics, list) else [])
        profile = {
            "participant": str(raw_profile.get("participant", "未明")).strip() or "未明",
            "age": str(raw_profile.get("age", "未明")).strip() or "未明",
            "identity_context": str(raw_profile.get("identity_context", "未明")).strip() or "未明",
            "characteristics": characteristics,
            "interview_summary": str(raw_profile.get("interview_summary", "")).strip()
                or fallback_profile["interview_summary"],
        }
        directions = []
        raw_directions = result.get("research_directions", [])
        if isinstance(raw_directions, list):
            for item in raw_directions:
                if not isinstance(item, dict):
                    continue
                ids = item.get("candidate_ids", [])
                valid_ids = ([str(value) for value in ids if str(value) in coding_by_id]
                             if isinstance(ids, list) else [])
                direction = str(item.get("direction", "")).strip()
                finding = str(item.get("possible_finding", "")).strip()
                opportunity = str(item.get("research_opportunity", "")).strip()
                shared_concept = str(item.get("shared_concept", "")).strip()
                concept_chars = re.sub(r"\s", "", shared_concept)
                direction_text = f"{direction} {finding} {opportunity}"
                supported_ids = [candidate_id for candidate_id in valid_ids if shared_concept and (
                    shared_concept in str(coding_by_id[candidate_id].get("code", ""))
                    or shared_concept in str(coding_by_id[candidate_id].get("evidence_quote", ""))
                    or shared_concept in str(coding_by_id[candidate_id].get("evidence_context", ""))
                )]
                if (direction and finding and opportunity and len(concept_chars) >= 2
                        and shared_concept in direction_text and supported_ids):
                    directions.append({
                        "direction": direction,
                        "possible_finding": finding,
                        "research_opportunity": opportunity,
                        "shared_concept": shared_concept,
                        "candidate_ids": supported_ids,
                        "caution": str(item.get("caution", "")).strip(),
                    })
        return {"profile": profile, "research_directions": directions, "coding_by_id": coding_by_id}
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        print(f"    警告：文本簡介／研究方向產生失敗：{exc}", flush=True)
        return {"profile": fallback_profile, "research_directions": [], "coding_by_id": coding_by_id}


def render_research_directions(synthesis: dict) -> str:
    """Render tentative findings with their exact evidence and code."""
    directions = synthesis.get("research_directions", [])
    coding_by_id = synthesis.get("coding_by_id", {})
    if not directions:
        ranked = sorted(coding_by_id.items(), key=lambda item: int(item[1].get("analytic_score", 0)), reverse=True)[:3]
        if not ranked:
            return "本篇沒有足夠且可可靠定位的 code 可形成研究方向；請研究者回到逐字稿檢查。"
        directions = [{
            "direction": "優先複核高分析分數的線索",
            "possible_finding": "以下是評分較高的初步 coding，可作為後續跨文本比較的起點。",
            "candidate_ids": [candidate_id for candidate_id, _coding in ranked],
            "research_opportunity": "比較這些高品質線索在其他受訪者中是否重複、相反或具有不同條件。",
            "caution": "這是模型篩選失敗時的分數回退，不代表研究結論。",
        }]
    blocks = []
    for number, direction in enumerate(directions, 1):
        lines = [
            f"方向 {number}｜{direction['direction']}",
            f"可能發現：{direction['possible_finding']}",
            f"研究機會：{direction.get('research_opportunity', '可進一步跨文本比較此線索。')}",
        ]
        for candidate_id in direction["candidate_ids"]:
            coding = coding_by_id.get(candidate_id)
            if coding:
                lines.extend([
                    f"相關 code：{coding.get('code', '')}",
                    f"語證：{coding.get('evidence_context') or coding.get('evidence_quote', '')}",
                ])
        if direction.get("caution"):
            lines.append(f"研究提醒：{direction['caution']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def analyze_chunk(
    guide: str,
    chunk: list[Segment],
    args: argparse.Namespace,
    label: str,
    depth: int = 0,
) -> tuple[list[dict], int]:
    """Analyze one chunk, recursively bisecting it when grounding fails.

    A local model can occasionally paraphrase evidence in a large prompt. A
    smaller retry usually restores exact copying. One bad subchunk must not
    discard valid results from the rest of a long interview.
    """
    respondent_only = bool(getattr(args, "respondent_only", False))
    allowed_ids = {s.id for s in chunk if not respondent_only or s.speaker == "受訪者"}
    if not allowed_ids:
        return [], 0
    system, prompt = build_prompt(
        guide,
        chunk,
        respondent_only=respondent_only,
        source_file=str(getattr(args, "source_file", "")),
        participant_role=str(getattr(args, "participant_role", "")),
    )
    last_error: RuntimeError | None = None
    for attempt in range(args.retries + 1):
        retry_prompt = prompt
        if attempt:
            retry_prompt += (
                "\n\n重要修正：上一次結果無法在原文定位。這次 evidence_quote 必須逐字複製自單一 segment，"
                "且 segment_id 必須是該證據所在段落；不要改字、摘要或跨段拼接。"
            )
        try:
            result = ollama_chat(args.host, args.model, system, retry_prompt, args.timeout)
            validated = validate_codings(result, chunk, allowed_ids=allowed_ids)
            chunk_by_id = {segment.id: segment for segment in chunk}
            attempted_substantive = any(
                isinstance(row, dict)
                and row.get("segment_id") in allowed_ids
                and row.get("segment_id") in chunk_by_id
                and not MINIMAL_RESPONSE.fullmatch(re.sub(r"^[+\-–—\s]+", "", chunk_by_id[row["segment_id"]].text.strip()))
                and not looks_like_interviewer_turn(chunk_by_id[row["segment_id"]])
                for row in result.get("codings", [])
            )
            if attempted_substantive and not validated:
                raise RuntimeError("模型回傳的 code 無法用原文證據定位")
            return validated, 0
        except RuntimeError as exc:
            last_error = exc
            # Connection failures affect every possible subchunk; fail fast so
            # the user gets the real infrastructure error.
            if "無法連線 Ollama" in str(exc):
                raise
            # Retrying the same oversized prompt wastes another full timeout.
            # Split it immediately; only a single-segment timeout is skipped.
            if "逾時" in str(exc):
                break
            if attempt < args.retries:
                time.sleep(2 ** attempt)

    assert last_error is not None
    if len(chunk) > 1:
        midpoint = len(chunk) // 2
        indent = "  " * depth
        failure_kind = "逾時" if "逾時" in str(last_error) else "定位失敗"
        print(
            f"    {indent}{label} {failure_kind}，改拆成 {midpoint} 句 + {len(chunk) - midpoint} 句重試",
            flush=True,
        )
        left_rows, left_skipped = analyze_chunk(guide, chunk[:midpoint], args, label + "A", depth + 1)
        right_rows, right_skipped = analyze_chunk(guide, chunk[midpoint:], args, label + "B", depth + 1)
        return left_rows + right_rows, left_skipped + right_skipped

    reason = "持續逾時" if "逾時" in str(last_error) else "無法可靠定位"
    print(f"    警告：略過{reason}的片段 {chunk[0].id}（原始第 {chunk[0].line} 行）：{last_error}", flush=True)
    return [], 1


def code_file(
    source: Path,
    guide: str,
    args: argparse.Namespace,
    destination: Path | None = None,
) -> tuple[Path, int]:
    segments = segment_transcript(read_document(source), getattr(args, "context_radius", 6))
    args.source_file = source.name
    args.participant_role = participant_role_from_filename(source, guide)
    if args.participant_role:
        print(f"  依研究指引與檔名判定主要受訪者角色：{args.participant_role}", flush=True)
    # Enforce respondent-only coding only when the source explicitly uses the
    # role label "受訪者:". A dash is merely a speaker marker and is never
    # sufficient evidence of research role.
    args.respondent_only = any(segment.speaker == "受訪者" for segment in segments)
    if args.respondent_only:
        print("  已偵測明確的「受訪者」角色標籤；其他角色段落只作為上下文。", flush=True)
    by_id = {segment.id: segment for segment in segments}
    all_codings: list[dict] = []
    skipped_segments = 0
    chunks = list(chunk_segments(segments, args.chunk_chars, args.chunk_segments))
    for index, chunk in enumerate(chunks, 1):
        print(f"  區塊 {index}/{len(chunks)}（{len(chunk)} 句）", flush=True)
        rows, skipped = analyze_chunk(guide, chunk, args, f"區塊 {index}")
        all_codings.extend(rows)
        skipped_segments += skipped

    initial_count = len(all_codings)
    eligible_count = sum(1 for row in all_codings if quality_eligible(row))
    if getattr(args, "focused_refinement", True) and all_codings:
        print(f"  聚焦精選：比較整份訪談的 {initial_count} 個候選 code…", flush=True)
        all_codings = refine_codings(guide, all_codings, args)
        print(f"  聚焦精選完成：保留 {len(all_codings)} 個具分析價值的 code。", flush=True)

    if all_codings:
        all_codings = refine_evidence_ranges(guide, all_codings, segments, args)
        print("  語證範圍校正完成：已依每筆 code 重選必要上下文。", flush=True)

    count_explanation = code_count_explanation(
        minimum=max(0, int(getattr(args, "min_codes", 10))),
        initial_count=initial_count,
        eligible_count=eligible_count,
        final_count=len(all_codings),
        skipped_segments=skipped_segments,
        focused_refinement=bool(getattr(args, "focused_refinement", True)),
    )
    focused_enabled = bool(getattr(args, "focused_refinement", True))
    configured_minimum = max(0, int(getattr(args, "min_codes", 10)))
    if not all_codings or (focused_enabled and len(all_codings) < configured_minimum):
        print(f"  🦭 Code 數量說明：{count_explanation}", flush=True)

    print("  文本簡介與研究方向：整理受訪者輪廓與可能發現…", flush=True)
    synthesis = generate_document_synthesis(guide, source, segments, all_codings, args)
    profile = synthesis["profile"]
    directions_text = render_research_directions(synthesis)
    print("  文本簡介與研究方向完成。", flush=True)

    destination = destination or output_path(source, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "row_type", "source_file", "segment_id", "line_number", "speaker",
            "context_before", "quote_verbatim", "context_after", "full_context",
            "supporting_segment_ids", "evidence_context", "evidence_quote",
            "code", "rationale", "why_this_code", "confidence", "analytic_score",
            "research_relevance", "behavior_pattern", "evidence_strength",
            "opportunity_potential", "inference_risk", "score_reason",
            "participant", "age", "identity_context", "characteristics",
            "interview_summary", "code_count_explanation", "research_directions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "row_type": "document_intro", "source_file": source.name,
            "participant": profile["participant"], "age": profile["age"],
            "identity_context": profile["identity_context"],
            "characteristics": "\n".join(f"• {item}" for item in profile["characteristics"]),
            "interview_summary": profile["interview_summary"],
            "code_count_explanation": count_explanation,
        })
        for coding in all_codings:
            segment = by_id[coding["segment_id"]]
            writer.writerow({
                "row_type": "coding",
                "source_file": source.name,
                "segment_id": segment.id,
                "line_number": segment.line,
                "speaker": segment.speaker,
                "context_before": segment.context_before,
                "quote_verbatim": segment.text,
                "context_after": segment.context_after,
                "full_context": segment.full_context,
                "supporting_segment_ids": "|".join(coding.get("supporting_segment_ids", [segment.id])),
                "evidence_context": coding.get("evidence_context", ""),
                "evidence_quote": coding["evidence_quote"],
                "code": coding["code"],
                "rationale": coding["rationale"],
                "why_this_code": coding["rationale"],
                "confidence": f'{coding["confidence"]:.2f}',
                "analytic_score": coding.get("analytic_score", ""),
                "research_relevance": coding.get("research_relevance", ""),
                "behavior_pattern": coding.get("behavior_pattern", ""),
                "evidence_strength": coding.get("evidence_strength", ""),
                "opportunity_potential": coding.get("opportunity_potential", ""),
                "inference_risk": coding.get("inference_risk", ""),
                "score_reason": coding.get("score_reason", ""),
            })
        writer.writerow({
            "row_type": "research_directions", "source_file": source.name,
            "research_directions": directions_text,
        })
    if skipped_segments:
        print(f"  注意：共有 {skipped_segments} 個片段因無法可靠定位而略過，其餘結果已保留。", flush=True)
    return destination, len(all_codings)


def collect_sources(inputs: list[Path], guide_path: Path, output_dir: Path) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found.extend(p for p in item.rglob("*") if p.suffix.lower() in SUPPORTED)
        elif item.suffix.lower() in SUPPORTED:
            found.append(item)
        else:
            print(f"略過不支援的檔案：{item}", file=sys.stderr)
    guide_resolved = guide_path.resolve()
    # CSV outputs are not in SUPPORTED, so input files do not need to be
    # excluded merely because they live beside (or inside) the output folder.
    # The old parent-folder exclusion incorrectly removed every transcript
    # when users chose the transcript folder itself as the output location.
    _ = output_dir
    unique = {
        p.resolve() for p in found
        if p.resolve() != guide_resolved
    }
    return sorted(unique, key=lambda p: str(p).lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Ollama 對訪談逐字稿做研究指引導向的 open coding")
    parser.add_argument("inputs", nargs="+", type=Path, help="逐字稿檔案或資料夾（可多個）")
    parser.add_argument("--guide", required=True, type=Path, help="研究背景／研究問題／code 關注方向文檔")
    parser.add_argument("--output-dir", type=Path, default=Path("coding_output"), help="CSV 輸出資料夾")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama 模型名稱")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama 位址")
    parser.add_argument("--chunk-chars", type=int, default=5000, help="每次送模型的約略字元上限")
    parser.add_argument("--chunk-segments", type=int, default=10, help="每次送模型的句數上限")
    parser.add_argument("--context-radius", type=int, default=6, help="每句前後可供模型判斷的上下文句數")
    parser.add_argument("--timeout", type=int, default=600, help="每區塊逾時秒數")
    parser.add_argument("--retries", type=int, default=2, help="模型格式錯誤或連線失敗重試次數")
    parser.add_argument("--overwrite", action="store_true", help="覆寫已存在的 CSV；預設跳過")
    parser.add_argument(
        "--keep-all-codes",
        action="store_false",
        dest="focused_refinement",
        help="不做整份訪談的聚焦精選，保留全部初始 codes",
    )
    parser.set_defaults(focused_refinement=True)
    parser.add_argument("--min-codes", type=int, default=10, help="聚焦精選後每份文檔至少保留的 code 數")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_chars < 500:
        print("--chunk-chars 不可小於 500", file=sys.stderr)
        return 2
    if not 1 <= args.chunk_segments <= 200:
        print("--chunk-segments 必須介於 1 到 200", file=sys.stderr)
        return 2
    if not 1 <= getattr(args, "context_radius", 6) <= 20:
        print("--context-radius 必須介於 1 到 20", file=sys.stderr)
        return 2
    if not 0 <= args.min_codes <= 10000:
        print("--min-codes 必須介於 0 到 10000", file=sys.stderr)
        return 2
    try:
        guide = read_document(args.guide).strip()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"無法讀取研究指引：{exc}", file=sys.stderr)
        return 2
    if not guide:
        print("研究指引是空白的", file=sys.stderr)
        return 2
    sources = collect_sources(args.inputs, args.guide, args.output_dir)
    if not sources:
        print("找不到支援的逐字稿檔案", file=sys.stderr)
        return 2

    failures = 0
    destinations = batch_output_paths(sources, args.output_dir)
    print(f"批次處理：共 {len(sources)} 份逐字稿")
    for source_index, source in enumerate(sources, 1):
        destination = destinations[source.resolve()]
        if destination.exists() and not args.overwrite:
            print(f"跳過 {source_index}/{len(sources)}（CSV 已存在）：{destination}")
            continue
        print(f"文件 {source_index}/{len(sources)}")
        print(f"分析：{source}")
        try:
            path, count = code_file(source, guide, args, destination=destination)
            print(f"完成：{path}（{count} 筆 coding）")
        except Exception as exc:  # Continue other interviews and report failure.
            failures += 1
            print(f"失敗：{source}：{exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
