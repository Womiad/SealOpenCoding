#!/usr/bin/env python3
"""Tkinter desktop interface for the local open-coding pipeline."""

from __future__ import annotations

import csv
import ctypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
SCRIPT = APP_DIR / "open_coding.py"
ICON = APP_DIR / "seal_open_coding_icon.png"
READER_ICON = APP_DIR / "seal_code_reader_icon.png"
APP_NAME = "海豹牌 Open Coding 工具"
APP_VERSION = "V1.7"
READER_DEFAULT_GEOMETRY = "940x600"
READER_MINIMUM_SIZE = (680, 440)
FILE_TYPES = [
    ("支援的文檔", "*.txt *.md *.docx"),
    ("文字檔", "*.txt"),
    ("Markdown", "*.md"),
    ("Word", "*.docx"),
    ("所有檔案", "*.*"),
]
CSV_FILE_TYPES = [("Coding CSV", "*.csv"), ("所有檔案", "*.*")]
_CPU_SAMPLE: tuple[int, int] | None = None
SEAL_THINKING_LINES = (
    "🦭 海豹正在敲 GPU 的門。",
    "🦭 海豹正在思考……",
    "🦭 海豹正在文本中游泳。",
    "🦭 海豹把前後文排排坐，看看事情怎麼發生。",
    "🦭 海豹正在句子之間撈重要線索。",
    "🦭 海豹用小鰭翻到下一段。",
    "🦭 海豹正在確認前因、行為和結果。",
    "🦭 海豹把太普通的 code 輕輕推回海裡。",
    "🦭 GPU 開門了，海豹把文本送進去。",
    "🦭 海豹正在找受訪者特別在意的事情。",
    "🦭 海豹把相似線索放在一起比較。",
    "🦭 海豹正在檢查這個 code 有沒有說清楚。",
    "🦭 海豹潛進上下文，找完整的前因後果。",
    "🦭 海豹正在避免把訪員的話算到受訪者頭上。",
    "🦭 海豹浮上來換口氣，等等繼續分析。",
    "🦭 海豹正在把理所當然的內容篩掉。",
    "🦭 海豹戴上小眼鏡，重新讀一次上下文。",
    "🦭 海豹正在跟 Ollama 討論這句話的意思。",
    "🦭 海豹發現一條線索，正在確認不是魚。",
    "🦭 海豹把原因放左邊，結果放右邊。",
    "🦭 海豹正在找這位講者真正特別的想法。",
    "🦭 海豹用尾巴圈出可能的行為模式。",
    "🦭 海豹正在確認這是不是訪員的問題。",
    "🦭 海豹把短短的 code 拉長成完整邏輯。",
    "🦭 海豹正在替每個 code 找原文證據。",
    "🦭 海豹在文字海裡看到一個因果關係。",
    "🦭 海豹正在比較這段和前面哪裡不同。",
    "🦭 海豹把重複的 code 疊在一起看。",
    "🦭 海豹正在判斷這是觀點、行為，還是普通日常。",
    "🦭 海豹請 GPU 再想清楚一點。",
    "🦭 海豹正在把完整上下文裝進小桶子。",
    "🦭 海豹游回前兩段，確認事情的起點。",
    "🦭 海豹游到後兩段，看看事情的結果。",
    "🦭 海豹正在避免腦補不存在的原因。",
    "🦭 海豹把有趣的線索留在岸上曬太陽。",
    "🦭 海豹正在做最後一次證據對照。",
)
SEAL_DOT_FRAMES = (".", "..", "...", "..", ".", "..", "...", "..")
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _filetime_value(value: object) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def cpu_usage_percent() -> float:
    """Return Windows system CPU usage without adding a dependency."""
    global _CPU_SAMPLE
    if os.name != "nt":
        return 0.0

    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    idle, kernel, user = FileTime(), FileTime(), FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return 0.0
    idle_value = _filetime_value(idle)
    total_value = _filetime_value(kernel) + _filetime_value(user)
    if _CPU_SAMPLE is None:
        _CPU_SAMPLE = (idle_value, total_value)
        return 0.0
    previous_idle, previous_total = _CPU_SAMPLE
    _CPU_SAMPLE = (idle_value, total_value)
    total_delta = max(1, total_value - previous_total)
    idle_delta = max(0, idle_value - previous_idle)
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def prevent_system_sleep(active: bool) -> None:
    """Keep Windows awake during an unattended batch; never force the display on."""
    if os.name == "nt":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def format_elapsed(seconds: float) -> str:
    """Format a monotonic duration for the UI and completion summary."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def batch_event_kind(message: str) -> str | None:
    """Classify only document-level terminal log lines."""
    text = message.strip()
    if text.startswith("完成："):
        return "completed"
    if text.startswith("失敗："):
        return "failed"
    if text.startswith("跳過") and "CSV 已存在" in text:
        return "skipped"
    return None


def system_usage_text() -> str:
    cpu = cpu_usage_percent()
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return f"CPU {cpu:.0f}%｜GPU 未偵測"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        utilization, used_mb, total_mb = [
            float(part.strip()) for part in result.stdout.strip().splitlines()[0].split(",")
        ]
        return f"CPU {cpu:.0f}%｜GPU {utilization:.0f}%｜VRAM {used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB"
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return f"CPU {cpu:.0f}%｜GPU 讀取中"


def seal_companion_for_log(message: str, rotation_index: int = 0) -> str | None:
    text = message.strip()
    if "區塊" in text and "/" in text:
        return SEAL_THINKING_LINES[rotation_index % len(SEAL_THINKING_LINES)]
    rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("分析：",), "🦭 海豹攤開這篇訪談，開始找重要線索。"),
        (("逾時", "改拆成"), "🦭⏱ 這段游得太久，海豹把它拆小再試，不會丟下整篇。"),
        (("定位失敗", "改拆成"), "🦭 這段有點滑，海豹把它拆小再仔細看。"),
        (("候選過少", "自動拆成"), "🦭🔎 這段線索少得不尋常，海豹拆小後再掃一次。"),
        (("警告：略過",), "🦭⚠ 海豹無法可靠確認這一小段，先保守略過。"),
        (("聚焦精選：",), "🦭 海豹把候選 code 排開，準備挑出真正重要的。"),
        (("初選",), "🦭 海豹把太細、無關和理所當然的 code 輕輕撥開。"),
        (("跨批比較",), "🦭 海豹把整篇線索放在一起，尋找反覆模式與特別觀點。"),
        (("聚焦精選完成",), "🦭 海豹留下比較有分析價值的 code。"),
        (("語證範圍校正",), "🦭 海豹沿著每筆 code 往前後游，只帶回真正相關的上下文。"),
        (("品質門檻後只有",), "🦭 海豹寧可少留幾個，也不拿無關 code 湊數。"),
        (("完成：",), "🦭✓ 海豹整理完成，CSV 已經安全放好了。"),
        (("失敗：",), "🦭❌ 海豹處理這篇時遇到問題，請查看上方訊息。"),
        (("跳過（CSV 已存在）",), "🦭 找到之前整理好的 CSV，這次先不重做。"),
    )
    for needles, companion in rules:
        if all(needle in text for needle in needles):
            return companion
    return None


def status_context_for_log(message: str) -> str | None:
    """Translate pipeline logs into a compact live-stage label."""
    text = message.strip()
    if text.startswith("分析："):
        return Path(text.split("：", 1)[1]).name
    if "初選" in text and "/" in text:
        match = re.search(r"初選\s+(\d+)/(\d+)", text)
        if match:
            return f"篩選階段 初選 {match.group(1)}/{match.group(2)}"
    if "跨批比較" in text or "最終跨訪談精選" in text:
        return "篩選階段 最終比較"
    if "聚焦精選：" in text:
        return "篩選階段 準備候選"
    if "聚焦精選完成" in text:
        return "篩選階段 完成"
    if "改用 code 分數" in text:
        return "篩選階段 分數回退"
    if "候選過少" in text and "自動拆成" in text:
        return "初始 Coding 候選過少重掃"
    if "語證範圍校正" in text:
        match = re.search(r"語證範圍校正\s+(\d+)/(\d+)", text)
        return f"語證校正 {match.group(1)}/{match.group(2)}" if match else "語證校正 完成"
    if "文本簡介與研究方向" in text:
        return "文本簡介與研究方向"
    if "區塊" in text and "/" in text:
        return text
    return None


def read_coding_csv(path: Path) -> list[dict[str, str]]:
    """Read a coding CSV without requiring Excel or another application."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV 沒有欄位標題")
        return [{key: value or "" for key, value in row.items() if key is not None} for row in reader]


def first_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def reader_context(row: dict[str, str], mode: str = "selected") -> tuple[str, str]:
    """Return the selected evidence or the wider audit context for a coding row."""
    original = first_value(row, "quote_verbatim", "original_text", "原始文字", "原文")
    selected = first_value(row, "evidence_context", "code_context")
    complete = first_value(row, "full_context", "完整上下文")
    if not complete:
        context_parts = [
            first_value(row, "context_before", "前文"),
            f"【目標片段】{original}" if original else "",
            first_value(row, "context_after", "後文"),
        ]
        complete = "\n".join(part for part in context_parts if part)
    selected = selected or complete
    complete = complete or selected
    if mode == "full":
        return "完整上下文（搜尋視窗／稽核用）", complete or "（舊版 CSV 沒有上下文欄位）"
    return "精選語證（與本 code 直接相關）", selected or "（舊版 CSV 沒有上下文欄位）"


class CodingResultReader(tk.Toplevel):
    """Card-style, one-coding-at-a-time CSV reader."""

    def __init__(self, parent: "OpenCodingGUI", paths: list[Path] | None = None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title(f"Seal Code Reader {APP_VERSION}（內建）")
        self.geometry(READER_DEFAULT_GEOMETRY)
        self.minsize(*READER_MINIMUM_SIZE)
        self.reader_icon: tk.PhotoImage | None = None
        self.header_icon: tk.PhotoImage | None = None
        if READER_ICON.is_file():
            try:
                self.reader_icon = tk.PhotoImage(file=str(READER_ICON))
                self.iconphoto(True, self.reader_icon)
            except tk.TclError:
                self.reader_icon = None
        elif parent.app_icon is not None:
            self.iconphoto(True, parent.app_icon)
        self.paths: list[Path] = []
        self.rows: list[dict[str, str]] = []
        self.index = 0
        self.file_var = tk.StringVar()
        self.position_var = tk.StringVar(value="尚未載入結果")
        self.meta_var = tk.StringVar()
        self.jump_var = tk.StringVar(value="1")
        self.context_mode_var = tk.StringVar(value="selected")
        self._build()
        self.bind("<Left>", lambda _event: self._previous())
        self.bind("<Right>", lambda _event: self._next())
        self.bind("<Control-o>", lambda _event: self.choose_csv())
        if paths:
            self.set_paths(paths)
        else:
            self.after(100, self.choose_csv)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        title_bar = ttk.Frame(outer)
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.columnconfigure(1, weight=1)
        if self.reader_icon is not None:
            self.header_icon = self.reader_icon.subsample(5, 5)
            ttk.Label(title_bar, image=self.header_icon).grid(row=0, column=0, rowspan=2, padx=(0, 10))
        ttk.Label(title_bar, text="Seal Code Reader", font=("Microsoft JhengHei UI", 18, "bold")).grid(row=0, column=1, sticky="sw")
        ttk.Label(
            title_bar,
            text="海豹 Open Coding 結果閱讀器 · 僅供閱讀，不會修改 CSV",
            foreground="#555555",
        ).grid(row=1, column=1, sticky="nw")

        file_bar = ttk.Frame(outer)
        file_bar.grid(row=2, column=0, sticky="ew", pady=(12, 10))
        file_bar.columnconfigure(1, weight=1)
        ttk.Label(file_bar, text="結果檔案").grid(row=0, column=0, sticky="w", padx=(0, 7))
        self.file_combo = ttk.Combobox(file_bar, textvariable=self.file_var, state="readonly")
        self.file_combo.grid(row=0, column=1, sticky="ew")
        self.file_combo.bind("<<ComboboxSelected>>", self._file_selected)
        ttk.Button(file_bar, text="開啟 CSV…  Ctrl+O", command=self.choose_csv).grid(row=0, column=2, padx=(8, 0))
        context_mode = ttk.Frame(file_bar)
        context_mode.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(context_mode, text="上下文顯示：").pack(side="left")
        ttk.Radiobutton(
            context_mode, text="精選語證", variable=self.context_mode_var,
            value="selected", command=self._render,
        ).pack(side="left", padx=(2, 10))
        ttk.Radiobutton(
            context_mode, text="完整上下文", variable=self.context_mode_var,
            value="full", command=self._render,
        ).pack(side="left")

        ttk.Label(outer, textvariable=self.meta_var, foreground="#555555").grid(row=3, column=0, sticky="w", pady=(0, 5))
        content = ttk.Panedwindow(outer, orient="horizontal")
        content.grid(row=4, column=0, sticky="nsew")
        left = ttk.Frame(content, padding=(0, 0, 6, 0))
        right = ttk.Frame(content, padding=(6, 0, 0, 0))
        content.add(left, weight=3)
        content.add(right, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        for row, weight in ((0, 2), (2, 2), (4, 3)):
            right.rowconfigure(row, weight=weight)

        self.context_box = ttk.LabelFrame(left, text="與本 code 相關的原文範圍", padding=8)
        self.context_box.grid(row=0, column=0, sticky="nsew")
        self.context_box.columnconfigure(0, weight=1)
        self.context_box.rowconfigure(0, weight=1)
        self.context_text = self._readonly_text(self.context_box, height=12, font=("Microsoft JhengHei UI", 11))

        self.original_box = ttk.LabelFrame(right, text="實際 Coding 片段", padding=8)
        self.original_box.grid(row=0, column=0, sticky="nsew")
        self.original_box.columnconfigure(0, weight=1)
        self.original_box.rowconfigure(0, weight=1)
        self.original_text = self._readonly_text(self.original_box, height=4, font=("Microsoft JhengHei UI", 11))

        self.code_heading = ttk.Label(right, text="Code（情境／前因 → 想法或行為 → 結果／意義）", font=("Microsoft JhengHei UI", 10, "bold"))
        self.code_heading.grid(row=1, column=0, sticky="w", pady=(7, 3))
        self.code_text = self._readonly_text(right, height=3, font=("Microsoft JhengHei UI", 12, "bold"), row=2)

        self.why_heading = ttk.Label(right, text="Why this code", font=("Microsoft JhengHei UI", 10, "bold"))
        self.why_heading.grid(row=3, column=0, sticky="w", pady=(7, 3))
        self.why_text = self._readonly_text(right, height=4, font=("Microsoft JhengHei UI", 11), row=4)

        nav = ttk.Frame(outer)
        nav.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        nav.columnconfigure(2, weight=1)
        self.previous_button = ttk.Button(nav, text="← 上一頁", command=self._previous)
        self.previous_button.grid(row=0, column=0)
        self.next_button = ttk.Button(nav, text="下一頁 →", command=self._next)
        self.next_button.grid(row=0, column=1, padx=(7, 14))
        ttk.Label(nav, textvariable=self.position_var).grid(row=0, column=2, sticky="w")
        ttk.Label(nav, text="跳到第").grid(row=0, column=3, padx=(8, 4))
        jump = ttk.Entry(nav, textvariable=self.jump_var, width=7)
        jump.grid(row=0, column=4)
        jump.bind("<Return>", lambda _event: self._jump())
        ttk.Button(nav, text="跳轉", command=self._jump).grid(row=0, column=5, padx=(5, 0))

    def _readonly_text(self, parent: tk.Misc, height: int, font: tuple, row: int = 0) -> tk.Text:
        widget = tk.Text(parent, height=height, wrap="word", font=font, padx=8, pady=7, state="disabled")
        widget.grid(row=row, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
        scrollbar.grid(row=row, column=1, sticky="ns")
        widget.configure(yscrollcommand=scrollbar.set)
        return widget

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def choose_csv(self) -> None:
        selected = filedialog.askopenfilenames(parent=self, title="選擇 Seal Coding 結果", filetypes=CSV_FILE_TYPES)
        if selected:
            self.set_paths([Path(path) for path in selected])
        elif not self.paths:
            self.position_var.set("請按「開啟 CSV」選擇結果")

    def set_paths(self, paths: list[Path]) -> None:
        unique = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".csv" and resolved not in seen:
                unique.append(resolved)
                seen.add(resolved)
        if not unique:
            messagebox.showwarning("沒有結果", "找不到可閱讀的 CSV。", parent=self)
            return
        self.paths = unique
        values = [str(path) for path in self.paths]
        self.file_combo.configure(values=values)
        self.file_var.set(values[0])
        self._load_path(self.paths[0])

    def _file_selected(self, _event: object = None) -> None:
        value = self.file_var.get()
        if value:
            self._load_path(Path(value))

    def _load_path(self, path: Path) -> None:
        try:
            rows = read_coding_csv(path)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            messagebox.showerror("無法閱讀 CSV", f"{path}\n\n{exc}", parent=self)
            return
        if not rows:
            messagebox.showinfo("沒有 Coding", f"這份 CSV 沒有 coding result：\n{path}", parent=self)
            self.rows = []
            self._show_empty()
            return
        self.rows = rows
        self.index = 0
        self._render()

    def _render(self) -> None:
        if not self.rows:
            self._show_empty()
            return
        row = self.rows[self.index]
        row_type = first_value(row, "row_type") or "coding"
        source = first_value(row, "source_file", "source", "來源文件")
        if row_type == "document_intro":
            participant = first_value(row, "participant") or "未明"
            age = first_value(row, "age") or "未明"
            identity = first_value(row, "identity_context") or "未明"
            traits = first_value(row, "characteristics") or "（未辨識出可可靠描述的特質）"
            summary = first_value(row, "interview_summary") or "（沒有文本簡介）"
            count_explanation = first_value(row, "code_count_explanation")
            self.context_box.configure(text="文本簡介")
            self.original_box.configure(text="受訪者輪廓與可觀察特質")
            self.code_heading.configure(text="訪談摘要")
            self.why_heading.configure(text="使用提醒")
            self.meta_var.set(f"來源：{source}  |  文本首頁")
            self._set_text(self.context_text, f"受訪者／角色：{participant}\n年齡：{age}\n身分與生活脈絡：{identity}")
            self._set_text(self.original_text, traits)
            self._set_text(self.code_text, summary)
            count_note = (
                f"🦭 海豹的 Code 數量說明\n{count_explanation}\n\n"
                if count_explanation else ""
            )
            self._set_text(
                self.why_text,
                count_note + "閱讀提醒\n此頁是模型依逐字稿整理的導讀；未明資料不應推測。請以原文與研究者判斷為準。",
            )
            self._finish_render()
            return
        if row_type == "research_directions":
            directions = first_value(row, "research_directions") or "（沒有足夠證據形成研究方向）"
            self.context_box.configure(text="海豹覺得的研究方向")
            self.original_box.configure(text="可能的質性發現、語證與 code")
            self.code_heading.configure(text="分析定位")
            self.why_heading.configure(text="研究提醒")
            self.meta_var.set(f"來源：{source}  |  研究方向尾頁")
            self._set_text(self.context_text, directions)
            self._set_text(self.original_text, "每個方向均應包含可能發現、相關 code 與逐字語證；請由研究者回到完整上下文驗證。")
            self._set_text(self.code_text, "供後續跨文本比較、memo writing 與人工複核使用。")
            self._set_text(self.why_text, "可能發現不是最終結論；需回到完整逐字稿、反例與其他受訪者資料驗證。")
            self._finish_render()
            return

        self.original_box.configure(text="實際 Coding 片段")
        self.code_heading.configure(text="Code（情境／前因 → 想法或行為 → 結果／意義）")
        self.why_heading.configure(text="Why this code")
        original = first_value(row, "quote_verbatim", "original_text", "原始文字", "原文")
        context_title, context_text = reader_context(row, self.context_mode_var.get())
        code = first_value(row, "code", "Code", "編碼")
        why = first_value(row, "why_this_code", "rationale", "coding_reason", "理由")
        segment = first_value(row, "segment_id", "segment", "段落")
        line = first_value(row, "line_number", "line", "原始行號")
        analytic_score = first_value(row, "analytic_score")
        confidence = first_value(row, "confidence")
        quality_parts = []
        for label, field in (
            ("相關", "research_relevance"), ("模式", "behavior_pattern"),
            ("語證", "evidence_strength"), ("機會", "opportunity_potential"),
            ("風險", "inference_risk"),
        ):
            value = first_value(row, field)
            if value:
                quality_parts.append(f"{label} {value}/4")
        metadata = []
        if source:
            metadata.append(f"來源：{source}")
        if segment:
            metadata.append(f"Segment：{segment}")
        if line:
            metadata.append(f"原始行號：{line}")
        if analytic_score:
            metadata.append(f"分析分數：{analytic_score}")
        if confidence:
            metadata.append(f"信心：{confidence}")
        if quality_parts:
            metadata.append("品質：" + "、".join(quality_parts))
        self.meta_var.set("  |  ".join(metadata))
        self.context_box.configure(text=context_title)
        self._set_text(self.context_text, context_text)
        self._set_text(self.original_text, original or "（CSV 沒有原始文字欄位）")
        self._set_text(self.code_text, code or "（CSV 沒有 code 欄位）")
        self._set_text(self.why_text, why or "（CSV 沒有 why_this_code／rationale 欄位）")
        self._finish_render()

    def _finish_render(self) -> None:
        self.position_var.set(f"第 {self.index + 1} / {len(self.rows)} 頁")
        self.jump_var.set(str(self.index + 1))
        self.previous_button.configure(state="normal" if self.index > 0 else "disabled")
        self.next_button.configure(state="normal" if self.index < len(self.rows) - 1 else "disabled")

    def _show_empty(self) -> None:
        self.meta_var.set("")
        self.position_var.set("沒有可顯示的 coding result")
        self._set_text(self.context_text, "")
        self._set_text(self.original_text, "")
        self._set_text(self.code_text, "")
        self._set_text(self.why_text, "")
        self.previous_button.configure(state="disabled")
        self.next_button.configure(state="disabled")

    def _previous(self) -> None:
        if self.rows and self.index > 0:
            self.index -= 1
            self._render()

    def _next(self) -> None:
        if self.rows and self.index < len(self.rows) - 1:
            self.index += 1
            self._render()

    def _jump(self) -> None:
        if not self.rows:
            return
        try:
            target = int(self.jump_var.get())
        except ValueError:
            messagebox.showwarning("編號無效", "請輸入整數編號。", parent=self)
            return
        if not 1 <= target <= len(self.rows):
            messagebox.showwarning("超出範圍", f"請輸入 1 到 {len(self.rows)}。", parent=self)
            return
        self.index = target - 1
        self._render()


class OpenCodingGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME}（Seal Open Coding）{APP_VERSION}")
        self.geometry("920x760")
        self.minsize(760, 640)
        self.app_icon: tk.PhotoImage | None = None
        self.header_icon: tk.PhotoImage | None = None
        self._load_brand_icon()
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.monitor_stop = threading.Event()
        self.seal_rotation_index = 0
        self.seal_animation_active = False
        self.seal_animation_frame = 0
        self.seal_status_context = ""
        self.output_snapshot: dict[Path, int] = {}
        self.last_result_files: list[Path] = []
        self.run_started_at: float | None = None
        self.last_elapsed_seconds = 0.0
        self.completed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.cancel_requested = False

        self.guide_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(APP_DIR / "coding_output"))
        self.model_var = tk.StringVar(value="qwen3:8b")
        self.host_var = tk.StringVar(value="http://127.0.0.1:11434")
        self.chunk_var = tk.StringVar(value="5000")
        self.chunk_segments_var = tk.StringVar(value="15")
        self.context_radius_var = tk.StringVar(value="12")
        self.min_context_segments_var = tk.StringVar(value="5")
        self.min_codes_var = tk.StringVar(value="10")
        self.overwrite_var = tk.BooleanVar(value=False)
        self.focused_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="請選擇研究指引與訪談文本")
        self.system_usage_var = tk.StringVar(value="CPU 讀取中｜GPU 讀取中")
        self.elapsed_var = tk.StringVar(value="已執行 00:00:00")

        self._build()
        threading.Thread(target=self._system_monitor_worker, daemon=True).start()
        self.after(420, self._animate_analysis_status)
        self.after(9000, self._rotate_seal_log)
        self.after(100, self._drain_events)
        self.after(500, self._update_elapsed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_brand_icon(self) -> None:
        """Use the supplied seal artwork for the window and in-app branding."""
        if not ICON.is_file():
            return
        try:
            self.app_icon = tk.PhotoImage(file=str(ICON))
            self.iconphoto(True, self.app_icon)
            # The supplied asset is 256 px; 64 px fits the header cleanly.
            self.header_icon = self.app_icon.subsample(4, 4)
        except tk.TclError:
            self.app_icon = None
            self.header_icon = None

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        outer.rowconfigure(7, weight=2)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.columnconfigure(1, weight=1)
        if self.header_icon is not None:
            ttk.Label(header, image=self.header_icon).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        ttk.Label(
            header,
            text=f"{APP_NAME}  |  Seal Open Coding",
            font=("Microsoft JhengHei UI", 16, "bold"),
        ).grid(row=0, column=1, sticky="sw")
        ttk.Label(header, text=f"{APP_VERSION} · 本機 LLM 訪談文本編碼工具", foreground="#555555").grid(row=1, column=1, sticky="nw")
        monitor = ttk.Frame(header)
        monitor.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(12, 0))
        tk.Label(
            monitor,
            textvariable=self.system_usage_var,
            fg="#176b45",
            font=("Microsoft JhengHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="e")
        tk.Label(
            monitor,
            textvariable=self.elapsed_var,
            fg="#555555",
            font=("Microsoft JhengHei UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="e")
        ttk.Label(
            outer,
            text="逐字稿會按句子分割，再依字數組成小段逐段分析；每份文檔各輸出一個 CSV。",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        source_box = ttk.LabelFrame(outer, text="1. 訪談文本（可多選）", padding=8)
        source_box.grid(row=2, column=0, sticky="nsew")
        source_box.columnconfigure(0, weight=1)
        source_box.rowconfigure(0, weight=1)
        self.sources = tk.Listbox(source_box, selectmode="extended", height=7)
        self.sources.grid(row=0, column=0, rowspan=4, sticky="nsew")
        scrollbar = ttk.Scrollbar(source_box, orient="vertical", command=self.sources.yview)
        scrollbar.grid(row=0, column=1, rowspan=4, sticky="ns")
        self.sources.configure(yscrollcommand=scrollbar.set)
        ttk.Button(source_box, text="加入檔案…", command=self._add_files).grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=(0, 4))
        ttk.Button(source_box, text="加入資料夾…", command=self._add_folder).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=4)
        ttk.Button(source_box, text="移除選取", command=self._remove_selected).grid(row=2, column=2, sticky="ew", padx=(8, 0), pady=4)
        ttk.Button(source_box, text="全部清除", command=lambda: self.sources.delete(0, "end")).grid(row=3, column=2, sticky="new", padx=(8, 0), pady=(4, 0))

        guide_box = ttk.LabelFrame(outer, text="2. 研究指引", padding=8)
        guide_box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        guide_box.columnconfigure(0, weight=1)
        ttk.Entry(guide_box, textvariable=self.guide_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(guide_box, text="選擇…", command=self._choose_guide).grid(row=0, column=1, padx=(8, 0))

        output_box = ttk.LabelFrame(outer, text="3. CSV 輸出資料夾", padding=8)
        output_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        output_box.columnconfigure(0, weight=1)
        ttk.Entry(output_box, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(output_box, text="選擇…", command=self._choose_output).grid(row=0, column=1, padx=(8, 0))

        settings = ttk.LabelFrame(outer, text="4. 模型與分段設定", padding=8)
        settings.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        for column in (1, 3, 5, 7):
            settings.columnconfigure(column, weight=1)
        ttk.Label(settings, text="Ollama 模型").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.model_var, width=18).grid(row=0, column=1, sticky="ew", padx=(5, 14))
        ttk.Label(settings, text="每批字數上限").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(settings, from_=1000, to=30000, increment=1000, textvariable=self.chunk_var, width=10).grid(row=0, column=3, sticky="ew", padx=(5, 14))
        ttk.Label(settings, text="每批句數").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(settings, from_=5, to=100, increment=5, textvariable=self.chunk_segments_var, width=8).grid(row=0, column=5, sticky="ew", padx=(5, 14))
        ttk.Label(settings, text="Ollama 位址").grid(row=0, column=6, sticky="w")
        ttk.Entry(settings, textvariable=self.host_var, width=22).grid(row=0, column=7, sticky="ew", padx=(5, 0))
        ttk.Label(settings, text="每篇至少 code 數").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Spinbox(settings, from_=0, to=1000, increment=1, textvariable=self.min_codes_var, width=10).grid(row=1, column=1, sticky="w", padx=(5, 14), pady=(7, 0))
        ttk.Label(settings, text="前後文搜尋上限").grid(row=1, column=2, sticky="w", pady=(7, 0))
        ttk.Spinbox(settings, from_=1, to=20, increment=1, textvariable=self.context_radius_var, width=8).grid(row=1, column=3, sticky="w", padx=(5, 14), pady=(7, 0))
        ttk.Label(settings, text="保底上下文句數").grid(row=1, column=4, sticky="w", pady=(7, 0))
        ttk.Spinbox(settings, from_=1, to=41, increment=1, textvariable=self.min_context_segments_var, width=8).grid(row=1, column=5, sticky="w", padx=(5, 14), pady=(7, 0))
        ttk.Label(settings, text="不足時在同一題內補相鄰發言", foreground="#555555").grid(row=1, column=6, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(
            settings,
            text="V1.7 會補抓基本資訊、重檢同脈絡 code，並依訪談順序呈現；候選過少仍會自動拆批重掃。",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=8, sticky="w", pady=(7, 0))

        action = ttk.Frame(outer)
        action.grid(row=6, column=0, sticky="ew", pady=10)
        action.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(action, text="開始 Open Coding", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(action, text="停止", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=8)
        self.read_button = ttk.Button(action, text="閱讀結果", command=self._read_results)
        self.read_button.grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(action, text="精選重要 code（建議）", variable=self.focused_var).grid(row=0, column=3, sticky="w", padx=(12, 8))
        ttk.Checkbutton(action, text="覆寫已存在的 CSV", variable=self.overwrite_var).grid(row=0, column=4, sticky="e")

        log_box = ttk.LabelFrame(outer, text="處理進度", padding=8)
        log_box.grid(row=7, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(1, weight=1)
        self.progress = ttk.Progressbar(log_box, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.log = tk.Text(log_box, height=10, wrap="word", state="disabled")
        self.log.grid(row=1, column=0, sticky="nsew")
        self.log.tag_configure("seal", foreground="#176b45", font=("Microsoft JhengHei UI", 10, "bold"))
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)
        ttk.Label(outer, textvariable=self.status_var).grid(row=8, column=0, sticky="w", pady=(6, 0))

    def _existing_sources(self) -> set[str]:
        return set(self.sources.get(0, "end"))

    def _add_paths(self, paths: tuple[str, ...] | list[str]) -> None:
        existing = self._existing_sources()
        for path in paths:
            normalized = str(Path(path).resolve())
            if normalized not in existing:
                self.sources.insert("end", normalized)
                existing.add(normalized)

    def _add_files(self) -> None:
        self._add_paths(list(filedialog.askopenfilenames(title="選擇訪談文本", filetypes=FILE_TYPES)))

    def _add_folder(self) -> None:
        path = filedialog.askdirectory(title="選擇包含逐字稿的資料夾")
        if path:
            self._add_paths([path])

    def _remove_selected(self) -> None:
        for index in reversed(self.sources.curselection()):
            self.sources.delete(index)

    def _choose_guide(self) -> None:
        path = filedialog.askopenfilename(title="選擇研究指引", filetypes=FILE_TYPES)
        if path:
            self.guide_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="選擇 CSV 輸出資料夾")
        if path:
            self.output_var.set(path)

    def _validate(self) -> tuple[list[str], int, int, int, int, int] | None:
        sources = list(self.sources.get(0, "end"))
        if not sources:
            messagebox.showwarning("尚未選擇", "請至少加入一份訪談文本或一個資料夾。")
            return None
        guide = Path(self.guide_var.get().strip())
        if not guide.is_file():
            messagebox.showwarning("研究指引無效", "請選擇一份存在的研究指引文檔。")
            return None
        if not self.output_var.get().strip():
            messagebox.showwarning("輸出位置無效", "請選擇 CSV 輸出資料夾。")
            return None
        if not self.model_var.get().strip():
            messagebox.showwarning("模型無效", "請輸入 Ollama 模型名稱。")
            return None
        try:
            chunk_chars = int(self.chunk_var.get())
            if not 500 <= chunk_chars <= 100000:
                raise ValueError
        except ValueError:
            messagebox.showwarning("分段設定無效", "每段字數必須是 500 到 100,000 的整數。")
            return None
        try:
            chunk_segments = int(self.chunk_segments_var.get())
            if not 1 <= chunk_segments <= 200:
                raise ValueError
        except ValueError:
            messagebox.showwarning("批次設定無效", "每批句數必須是 1 到 200 的整數。")
            return None
        try:
            min_codes = int(self.min_codes_var.get())
            if not 0 <= min_codes <= 10000:
                raise ValueError
        except ValueError:
            messagebox.showwarning("最低數量無效", "每篇至少 code 數必須是 0 到 10,000 的整數。")
            return None
        try:
            context_radius = int(self.context_radius_var.get())
            if not 1 <= context_radius <= 20:
                raise ValueError
        except ValueError:
            messagebox.showwarning("上下文設定無效", "前後文搜尋上限必須是 1 到 20 的整數。")
            return None
        try:
            min_context_segments = int(self.min_context_segments_var.get())
            if not 1 <= min_context_segments <= 2 * context_radius + 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "保底上下文設定無效",
                f"保底上下文句數必須是 1 到 {2 * context_radius + 1} 的整數。",
            )
            return None
        return sources, chunk_chars, chunk_segments, min_codes, context_radius, min_context_segments

    def _start(self) -> None:
        validated = self._validate()
        if validated is None:
            return
        sources, chunk_chars, chunk_segments, min_codes, context_radius, min_context_segments = validated
        command = [
            sys.executable, "-u", str(SCRIPT), *sources,
            "--guide", self.guide_var.get().strip(),
            "--output-dir", self.output_var.get().strip(),
            "--model", self.model_var.get().strip(),
            "--host", self.host_var.get().strip(),
            "--chunk-chars", str(chunk_chars),
            "--chunk-segments", str(chunk_segments),
            "--context-radius", str(context_radius),
            "--min-context-segments", str(min_context_segments),
            "--min-codes", str(min_codes),
        ]
        if self.overwrite_var.get():
            command.append("--overwrite")
        if not self.focused_var.get():
            command.append("--keep-all-codes")
        self.output_snapshot = self._snapshot_csvs()
        self.last_result_files = []
        self.seal_animation_active = True
        self.seal_rotation_index = 0
        self.seal_animation_frame = 0
        self.seal_status_context = ""
        self.run_started_at = time.monotonic()
        self.last_elapsed_seconds = 0.0
        self.completed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.cancel_requested = False
        self.elapsed_var.set("已執行 00:00:00")
        prevent_system_sleep(True)
        self._append_log("開始執行。訪談內容只會傳給指定的本機 Ollama。\n")
        self._append_log("🦭 海豹把研究指引放在旁邊，準備逐篇閱讀。\n", seal=True)
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.status_var.set("分析中…")
        threading.Thread(target=self._run_process, args=(command,), daemon=True).start()

    def _snapshot_csvs(self) -> dict[Path, int]:
        folder = Path(self.output_var.get().strip())
        if not folder.is_dir():
            return {}
        snapshot: dict[Path, int] = {}
        for path in folder.glob("*_open_coding.csv"):
            try:
                snapshot[path.resolve()] = path.stat().st_mtime_ns
            except OSError:
                continue
        return snapshot

    def _new_or_updated_csvs(self) -> list[Path]:
        current = self._snapshot_csvs()
        changed = [path for path, modified in current.items() if self.output_snapshot.get(path) != modified]
        return sorted(changed, key=lambda path: path.stat().st_mtime_ns, reverse=True)

    def _read_results(self) -> None:
        paths = [path for path in self.last_result_files if path.is_file()]
        CodingResultReader(self, paths or None)

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("log", line))
            return_code = self.process.wait()
            self.events.put(("done", return_code))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.process = None

    def _cancel(self) -> None:
        if self.process and self.process.poll() is None:
            self.cancel_requested = True
            self.process.terminate()
            self.seal_animation_active = False
            self._append_log("已要求停止；正在處理的該份文檔不會產生半成品 CSV。\n")
            self.status_var.set("正在停止…")

    def _append_log(self, text: str, seal: bool = False) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, "seal" if seal else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _system_monitor_worker(self) -> None:
        while not self.monitor_stop.is_set():
            self.events.put(("system_usage", system_usage_text()))
            self.monitor_stop.wait(2.0)

    def _next_seal_thinking_line(self) -> str:
        line = SEAL_THINKING_LINES[self.seal_rotation_index % len(SEAL_THINKING_LINES)]
        self.seal_rotation_index += 1
        return line

    def _rotate_seal_log(self) -> None:
        if self.seal_animation_active:
            self._append_log(self._next_seal_thinking_line() + "\n", seal=True)
        self.after(9000, self._rotate_seal_log)

    def _animate_analysis_status(self) -> None:
        if self.seal_animation_active:
            dots = SEAL_DOT_FRAMES[self.seal_animation_frame % len(SEAL_DOT_FRAMES)]
            self.seal_animation_frame += 1
            context = f"｜{self.seal_status_context}" if self.seal_status_context else ""
            self.status_var.set(f"🦭 海豹正在分析{dots}{context}")
        self.after(420, self._animate_analysis_status)

    def _update_elapsed(self) -> None:
        if self.run_started_at is not None:
            self.last_elapsed_seconds = time.monotonic() - self.run_started_at
            self.elapsed_var.set(f"已執行 {format_elapsed(self.last_elapsed_seconds)}")
        self.after(500, self._update_elapsed)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    message = str(value)
                    self._append_log(message)
                    stripped = message.strip()
                    batch_event = batch_event_kind(stripped)
                    if batch_event == "completed":
                        self.completed_count += 1
                    elif batch_event == "skipped":
                        self.skipped_count += 1
                    elif batch_event == "failed":
                        self.failed_count += 1
                    stage_context = status_context_for_log(stripped)
                    if stage_context:
                        self.seal_status_context = stage_context
                    if "區塊" in message and "/" in message:
                        companion = self._next_seal_thinking_line()
                    else:
                        companion = seal_companion_for_log(message, self.seal_rotation_index)
                    if companion:
                        self._append_log(companion + "\n", seal=True)
                elif kind == "system_usage":
                    self.system_usage_var.set(str(value))
                elif kind == "done":
                    self._finish(int(value))
                elif kind == "error":
                    self._append_log(f"GUI 執行錯誤：{value}\n")
                    self.failed_count += 1
                    self._finish(1)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _finish(self, return_code: int) -> None:
        self.seal_animation_active = False
        if self.run_started_at is not None:
            self.last_elapsed_seconds = time.monotonic() - self.run_started_at
            self.run_started_at = None
        elapsed = format_elapsed(self.last_elapsed_seconds)
        self.elapsed_var.set(f"已執行 {elapsed}")
        prevent_system_sleep(False)
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.last_result_files = self._new_or_updated_csvs()
        summary = (
            f"執行時間：{elapsed}\n"
            f"成功完成：{self.completed_count} 份\n"
            f"跳過既有結果：{self.skipped_count} 份\n"
            f"失敗：{self.failed_count} 份"
        )
        if return_code == 0:
            self.status_var.set(f"全部完成｜{elapsed}｜完成 {self.completed_count} 份")
            message = summary + "\n\n按主畫面的「閱讀結果」即可逐頁查看。"
            messagebox.showinfo("Open Coding 完成", message)
        elif self.cancel_requested:
            self.status_var.set(f"已停止｜執行 {elapsed}｜完成 {self.completed_count} 份")
        else:
            self.status_var.set(f"批次結束｜{elapsed}｜完成 {self.completed_count} 份、失敗 {self.failed_count} 份")
            messagebox.showwarning("Open Coding 批次結束", summary + "\n\n部分文檔失敗，請查看處理進度；其他完成結果仍可閱讀。")

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("仍在執行", "分析仍在進行。要停止並關閉嗎？"):
                return
            self.process.terminate()
        self.monitor_stop.set()
        self.seal_animation_active = False
        prevent_system_sleep(False)
        self.destroy()


if __name__ == "__main__":
    OpenCodingGUI().mainloop()
