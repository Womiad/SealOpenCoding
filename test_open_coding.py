import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import open_coding
import open_coding_gui


class OpenCodingTests(unittest.TestCase):
    def test_reader_loads_utf8_bom_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_file", "code"])
                writer.writeheader()
                writer.writerow({"source_file": "fixture.txt", "code": "測試編碼"})
            self.assertEqual(open_coding_gui.read_coding_csv(path)[0]["code"], "測試編碼")

    def test_segmentation_preserves_location_speaker_and_context(self):
        segments = open_coding.segment_transcript("提問者：測試問題。\n參與者：測試陳述甲。測試陳述乙！")
        self.assertEqual([item.text for item in segments], ["測試問題。", "測試陳述甲。", "測試陳述乙！"])
        self.assertEqual(segments[1].speaker, "參與者")
        self.assertEqual(segments[1].line, 2)
        self.assertIn("S000001", segments[1].context_before)
        self.assertIn("S000003", segments[1].context_after)

    def test_context_radius_is_only_a_reading_limit(self):
        text = "\n".join(f"參與者：合成測試句{i}。" for i in range(1, 16))
        target = open_coding.segment_transcript(text, context_radius=6)[7]
        self.assertEqual(target.context_before.count("【S"), 6)
        self.assertEqual(target.context_after.count("【S"), 6)

    def test_plus_turn_marker_is_preserved_without_assuming_role(self):
        segments = open_coding.segment_transcript("+ 這是帶有加號的合成發言。")
        self.assertEqual(segments[0].speaker, "講者（+標記）")
        self.assertEqual(segments[0].text, "這是帶有加號的合成發言。")

    def test_timestamp_colon_is_not_parsed_as_speaker(self):
        segments = open_coding.segment_transcript("2026/1/2 13:05-14:11")
        self.assertEqual(segments[0].speaker, "")
        self.assertEqual(segments[0].text, "2026/1/2 13:05-14:11")

    def test_colon_after_sentence_is_not_parsed_as_speaker(self):
        source = "這是一段較長的合成說明。接著顯示時間12:34"
        segments = open_coding.segment_transcript(source)
        self.assertEqual(segments[0].speaker, "")
        self.assertIn("合成說明", segments[0].text)

    def test_selected_evidence_context_is_source_grounded_and_flexible(self):
        text = "\n".join([
            "提問者：無關測試句。", "提問者：合成情境甲。", "參與者：合成行動乙。",
            "參與者：合成反應丙。", "提問者：合成追問丁。", "參與者：合成評價戊。",
            "參與者：合成條件己並說明後續選擇。", "提問者：切換測試主題。",
        ])
        segments = open_coding.segment_transcript(text, context_radius=6)
        result = {"codings": [{
            "segment_id": "S000007",
            "supporting_segment_ids": ["S000002", "S000003", "S000004", "S000005", "S000006", "S000007", "S999999"],
            "evidence_quote": "合成條件己並說明後續選擇",
            "code": "在合成測試情境下，參與者形成一項條件式選擇。",
            "rationale": "選定的合成測試句共同支持此結構。",
            "confidence": 0.9,
        }]}
        cleaned = open_coding.validate_codings(result, segments)
        self.assertEqual(cleaned[0]["supporting_segment_ids"], [f"S{i:06d}" for i in range(2, 8)])
        self.assertNotIn("無關測試句", cleaned[0]["evidence_context"])
        self.assertNotIn("切換測試主題", cleaned[0]["evidence_context"])

    def test_hallucinated_evidence_is_rejected(self):
        segments = open_coding.segment_transcript("參與者：來源內只有合成字串甲。")
        result = {"codings": [{
            "segment_id": "S000001", "evidence_quote": "不存在字串乙",
            "code": "錯誤測試 code", "rationale": "錯誤測試理由", "confidence": 0.9,
        }]}
        self.assertEqual(open_coding.validate_codings(result, segments), [])

    def test_evidence_refinement_moves_primary_and_keeps_long_relevant_episode(self):
        text = "\n".join([
            "提問者：請先想像一個合成情境。",
            "提問者：你會怎麼處理這個合成事件？",
            "+ 我會先執行合成步驟甲。",
            "+ 接著觀察合成結果乙。",
            "提問者：這會改變你的選擇嗎？",
            "+ 因為結果乙符合條件，所以我會採用方案丙。",
        ])
        segments = open_coding.segment_transcript(text, context_radius=6)
        primary = segments[1]
        codings = [{
            "segment_id": primary.id,
            "supporting_segment_ids": [primary.id],
            "evidence_context": primary.full_context,
            "evidence_quote": primary.text,
            "code": "在合成事件中，參與者依觀察結果形成條件式選擇。",
            "rationale": "完整事件支持此 code。",
            "full_context": primary.full_context,
            "context_before": primary.context_before,
            "context_after": primary.context_after,
        }]
        model_result = {"contexts": [{
            "candidate_id": "C00001",
            "primary_segment_id": "S000006",
            "supporting_segment_ids": [f"S{i:06d}" for i in range(1, 7)],
            "reason": "六句共同構成情境、行動、觀察與選擇。",
        }]}
        args = Namespace(host="local", model="test", timeout=10)
        with patch.object(open_coding, "ollama_chat", return_value=model_result):
            rows = open_coding.refine_evidence_ranges("合成研究規則", codings, segments, args)
        self.assertEqual(rows[0]["segment_id"], "S000006")
        self.assertEqual(len(rows[0]["supporting_segment_ids"]), 6)
        self.assertIn("合成步驟甲", rows[0]["evidence_context"])
        self.assertEqual(rows[0]["evidence_quote"], segments[5].text)

    def test_invalid_evidence_refinement_preserves_existing_context(self):
        segments = open_coding.segment_transcript("參與者：這是一句完整的合成經驗陳述。")
        original = {
            "segment_id": "S000001", "supporting_segment_ids": ["S000001"],
            "evidence_context": segments[0].full_context, "evidence_quote": segments[0].text,
            "code": "合成 code", "rationale": "合成理由", "full_context": segments[0].full_context,
        }
        result = {"contexts": [{
            "candidate_id": "C00001", "primary_segment_id": "S999999",
            "supporting_segment_ids": ["S999999"], "reason": "無效測試",
        }]}
        args = Namespace(host="local", model="test", timeout=10)
        with patch.object(open_coding, "ollama_chat", return_value=result):
            rows = open_coding.refine_evidence_ranges("合成研究規則", [original], segments, args)
        self.assertEqual(rows[0], original)

    def test_short_agreement_is_rejected(self):
        segments = open_coding.segment_transcript("提問者：合成問題？\n參與者：+對")
        result = {"codings": [{
            "segment_id": "S000002", "evidence_quote": "+對",
            "code": "由短答過度推論", "rationale": "測試", "confidence": 0.9,
        }]}
        self.assertEqual(open_coding.validate_codings(result, segments), [])

    def test_generic_filename_role_rule_comes_from_guide(self):
        guide = "檔名以 A 開頭代表第一類參與者。\n檔名以 B 開頭代表第二類參與者。"
        self.assertEqual(open_coding.participant_role_from_filename(Path("A_fixture.txt"), guide), "第一類參與者")
        self.assertEqual(open_coding.participant_role_from_filename(Path("B_fixture.txt"), guide), "第二類參與者")
        self.assertEqual(open_coding.participant_role_from_filename(Path("A_fixture.txt"), "沒有角色規則"), "")

    def test_quality_gate_uses_dimensions(self):
        segment = open_coding.Segment("S000001", 1, "參與者", "這是一個足夠長的合成測試陳述。")
        strong = open_coding.quality_dimensions({
            "research_relevance": 4, "behavior_pattern": 3, "evidence_strength": 4,
            "opportunity_potential": 2, "inference_risk": 0,
        }, segment)
        weak = open_coding.quality_dimensions({
            "research_relevance": 1, "behavior_pattern": 4, "evidence_strength": 4,
            "opportunity_potential": 0, "inference_risk": 0,
        }, segment)
        self.assertGreater(strong["analytic_score"], weak["analytic_score"])
        self.assertFalse(open_coding.quality_eligible(weak))

    def test_zero_minimum_explains_zero_without_padding(self):
        explanation = open_coding.code_count_explanation(0, 0, 0, 0, 0, True)
        self.assertIn("允許輸出 0 個", explanation)

    def test_below_minimum_explains_gate_and_skips(self):
        explanation = open_coding.code_count_explanation(10, 7, 4, 4, 2, True)
        self.assertIn("設定至少 10 個", explanation)
        self.assertIn("2 個片段", explanation)

    def test_timeout_splits_chunk_and_preserves_other_result(self):
        segments = open_coding.segment_transcript("參與者：合成片段甲。\n參與者：合成片段乙包含有效動作。")
        args = Namespace(host="local", model="test", timeout=1, retries=0, min_codes=0)

        def fake_chat(_host, _model, _system, prompt, _timeout):
            if prompt.count('"coding_allowed"') > 1 or '"segment_id": "S000001"' in prompt:
                raise RuntimeError("模型請求逾時（1 秒）")
            return {"codings": [{
                "segment_id": "S000002", "evidence_quote": "合成片段乙包含有效動作",
                "code": "在合成情境下，參與者執行有效動作。", "rationale": "來源直接支持。",
                "confidence": 0.9,
            }]}

        with patch.object(open_coding, "ollama_chat", side_effect=fake_chat):
            rows, skipped = open_coding.analyze_chunk("合成研究規則", segments, args, "測試區塊")
        self.assertEqual(skipped, 1)
        self.assertEqual([row["segment_id"] for row in rows], ["S000002"])

    def test_code_file_writes_source_grounded_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.txt"
            source.write_text("參與者：合成來源句包含可定位證據。", encoding="utf-8")
            args = Namespace(
                chunk_chars=9000, chunk_segments=30, context_radius=6,
                host="local", model="test", timeout=10, retries=0,
                output_dir=root / "out", min_codes=0, focused_refinement=True,
            )
            coding_result = {"codings": [{
                "segment_id": "S000001", "evidence_quote": "合成來源句包含可定位證據",
                "code": "在合成情境下，參與者提供可定位陳述。", "rationale": "來源直接支持。",
                "confidence": 0.9,
            }]}
            synthesis_result = {
                "profile": {"participant": "參與者", "age": "未明", "identity_context": "未明",
                            "characteristics": [], "interview_summary": "合成測試摘要。"},
                "research_directions": [],
            }
            context_result = {"contexts": [{
                "candidate_id": "C00001", "primary_segment_id": "S000001",
                "supporting_segment_ids": ["S000001"], "reason": "單句已完整。",
            }]}
            with patch.object(open_coding, "ollama_chat", side_effect=[coding_result, context_result, synthesis_result]):
                output, count = open_coding.code_file(source, "合成研究規則", args)
            self.assertEqual(count, 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["quote_verbatim"], "合成來源句包含可定位證據。")
            self.assertEqual(rows[1]["supporting_segment_ids"], "S000001")

    def test_batch_duplicate_names_have_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "fixture.txt"
            second = root / "b" / "fixture.txt"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("甲", encoding="utf-8")
            second.write_text("乙", encoding="utf-8")
            outputs = open_coding.batch_output_paths([first, second], root / "out")
            self.assertNotEqual(outputs[first.resolve()], outputs[second.resolve()])

    def test_seal_status_helpers(self):
        self.assertGreaterEqual(len(open_coding_gui.SEAL_THINKING_LINES), 30)
        self.assertEqual(open_coding_gui.format_elapsed(3661.9), "01:01:01")
        self.assertEqual(open_coding_gui.status_context_for_log("初選 2/5：20 個候選"), "篩選階段 初選 2/5")
        self.assertEqual(open_coding_gui.status_context_for_log("語證範圍校正 2/3：8 個 code"), "語證校正 2/3")


if __name__ == "__main__":
    unittest.main()
