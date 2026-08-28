# Seal 評估套件（2026-08-26）

三個部分：Part A 易用性測試、Part B 本機／雲端 LLM 效能比較（合成資料 + 埋點）、Part C 驗證行為實驗。
先讀《Seal_評估計畫書.docx》第 1–2 節，再看《Seal_易用性測試執行手冊.docx》。

```
Seal_評估計畫書.docx            設計、理由、「用聊天機器人生成資料比較本機／雲端」的可行性分析、合成資料生成 prompt
Seal_易用性測試執行手冊.docx     主持人腳本、任務卡 T0–T7、同意書、前測問卷、SUS、量表、訪談大綱、Part C 材料製作
Seal_評估記錄表.xlsx            所有記錄與計分（黃底藍字＝要填；灰底「範例」列使用前刪除）
研究指引_音效與環境音.txt        T4 用的第二份研究指引（另一份已複製到 test_kit/）
seal_cloud_proxy.py             Part B：在 127.0.0.1 模仿 Ollama API、轉送到雲端（只送合成資料；需 SEAL_EVAL_SYNTHETIC_ONLY=1）
seal_eval_collect.py            Part B：收集 results/<條件>/run<N>/*.csv → summary / codes_long / planted_autocheck / stability
seal_make_audit_set.py          Part C：從一份結果 CSV 產生稽核版 A（顯示理由）與 B（隱藏理由）
seal_dash_to_roles.py           把 Seal STT 的「-」講者格式轉成「訪員：／受訪者：」，再交給 Seal Open Coding
paper/
  mainsinglefile_revised.tex    修訂版論文（可直接編譯；改動處附 %% [Revision note]）
  mainsinglefile_revised.pdf    修訂版編譯結果（4 頁）
  mainsinglefile_changes.diff   與原稿的逐行差異
  mainsinglefile_修改清單.docx   每處改動的理由、camera-ready 前必做事項、已核對的技術敘述
```

三支 Python 腳本只用標準函式庫，Python 3.10+ 即可執行；`python 檔名.py --help` 看參數。
