import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import open_coding
import open_coding_gui


class OpenCodingTests(unittest.TestCase):
    def test_built_in_reader_uses_compact_horizontal_window(self):
        width, height = map(int, open_coding_gui.READER_DEFAULT_GEOMETRY.split("x"))
        self.assertLessEqual(height, 600)
        self.assertLessEqual(open_coding_gui.READER_MINIMUM_SIZE[1], 440)
        self.assertGreater(width, height)

    def test_cli_defaults_use_fifteen_sentence_batches(self):
        with patch.object(sys, "argv", [
            "open_coding.py", "fixture.txt", "--guide", "guide.txt",
        ]):
            args = open_coding.parse_args()
        self.assertEqual(args.chunk_chars, 5000)
        self.assertEqual(args.chunk_segments, 15)

    def test_reader_loads_utf8_bom_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_file", "code"])
                writer.writeheader()
                writer.writerow({"source_file": "fixture.txt", "code": "測試編碼"})
            self.assertEqual(open_coding_gui.read_coding_csv(path)[0]["code"], "測試編碼")

    def test_reader_can_switch_between_selected_and_full_context(self):
        row = {
            "quote_verbatim": "目標回答",
            "evidence_context": "主題問題\n目標回答",
            "full_context": "較早的討論\n主題問題\n目標回答\n後續追問",
        }
        selected_title, selected = open_coding_gui.reader_context(row)
        full_title, full = open_coding_gui.reader_context(row, "full")
        self.assertIn("精選語證", selected_title)
        self.assertEqual(selected, "主題問題\n目標回答")
        self.assertIn("完整上下文", full_title)
        self.assertIn("較早的討論", full)
        self.assertIn("後續追問", full)

    def test_research_direction_recovery_groups_multiple_codes(self):
        segments = open_coding.segment_transcript(
            "參與者：遇到任務壓力時會先暫停操作。\n參與者：壓力大時原本只想休息一下，最後卻放棄任務。"
        )
        codings = [
            {
                "segment_id": segments[0].id, "evidence_quote": segments[0].text,
                "evidence_context": segments[0].text,
                "code": "任務壓力觸發暫停操作作為因應", "rationale": "直接描述觸發條件。",
            },
            {
                "segment_id": segments[1].id, "evidence_quote": segments[1].text,
                "evidence_context": segments[1].text,
                "code": "任務壓力下從短暫休息走向放棄", "rationale": "描述行為發展與結果。",
            },
        ]
        profile_result = {
            "participant": {"value": "參與者", "segment_id": "S000001",
                            "evidence_quote": "遇到任務壓力"},
            "age": {"value": "未明", "segment_id": "", "evidence_quote": ""},
            "identity_facts": [{"fact": "正在執行任務", "segment_id": "S000001",
                                "evidence_quote": "任務壓力"}],
            "characteristics": [{"trait": "遇到壓力會先暫停", "segment_id": "S000001",
                                 "evidence_quote": "會先暫停操作"}],
            "interview_summary": "合成摘要。",
        }
        first_result = {"research_directions": []}
        recovery_result = {"research_directions": [{
            "direction": "任務壓力下的中斷歷程",
            "possible_finding": "任務壓力不只觸發暫停，也可能讓短暫休息逐步走向放棄。",
            "research_opportunity": "比較不同任務壓力如何改變中斷歷程與可介入時點。",
            "shared_concept": "任務中斷歷程",
            "evidence_links": [
                {"candidate_id": "C00001", "connection": "呈現壓力先觸發暫停操作。",
                 "anchor_quote": "任務壓力觸發暫停操作"},
                {"candidate_id": "C00002", "connection": "呈現暫停之後可能走向放棄。",
                 "anchor_quote": "從短暫休息走向放棄"},
            ],
            "caution": "仍需比較非壓力情境。",
        }]}
        args = Namespace(host="local", model="test", timeout=10)
        with patch.object(
            open_coding, "ollama_chat",
            side_effect=[profile_result, first_result, recovery_result],
        ) as chat:
            synthesis = open_coding.generate_document_synthesis(
                "研究任務壓力與中斷行為", Path("fixture.txt"), segments, codings, args,
            )
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(synthesis["profile"]["participant"], "參與者")
        self.assertEqual(synthesis["profile"]["identity_context"], "正在執行任務")
        self.assertEqual(len(synthesis["research_directions"]), 1)
        self.assertEqual(
            synthesis["research_directions"][0]["candidate_ids"],
            ["C00001", "C00002"],
        )

    def test_opening_profile_extracts_only_grounded_basic_information(self):
        segments = open_coding.segment_transcript(
            "訪員：請問你今年幾歲？\n參與者：我今年四十二歲，在合成單位工作。"
        )
        result = {
            "participant": {"value": "受訪者", "segment_id": "S000002",
                            "evidence_quote": "我今年四十二歲"},
            "age": {"value": "42 歲", "segment_id": "S000002",
                    "evidence_quote": "四十二歲"},
            "identity_facts": [
                {"fact": "在合成單位工作", "segment_id": "S000002",
                 "evidence_quote": "在合成單位工作"},
                {"fact": "不存在的家庭資訊", "segment_id": "S000002",
                 "evidence_quote": "沒有出現在原文"},
            ],
            "characteristics": [], "interview_summary": "開頭詢問基本資訊。",
        }
        args = Namespace(host="local", model="test", timeout=10, participant_role="")
        with patch.object(open_coding, "ollama_chat", return_value=result):
            profile = open_coding.generate_document_profile(
                "合成研究規則", Path("fixture.txt"), segments, [], args,
            )
        self.assertEqual(profile["age"], "42 歲")
        self.assertEqual(profile["identity_context"], "在合成單位工作")
        self.assertNotIn("家庭", profile["identity_context"])

    def test_research_direction_rejects_single_code_restatement(self):
        coding_by_id = {
            "C00001": {"code": "任務壓力觸發暫停", "evidence_quote": "任務壓力大就先暫停"},
            "C00002": {"code": "任務壓力下放棄", "evidence_quote": "任務壓力大就會放棄"},
        }
        directions, reasons = open_coding._validate_research_directions([{
            "direction": "任務壓力與中斷", "possible_finding": "任務壓力可能觸發暫停。",
            "research_opportunity": "追問任務壓力來源。", "shared_concept": "中斷歷程",
            "evidence_links": [{
                "candidate_id": "C00001", "connection": "呈現暫停條件。",
                "anchor_quote": "任務壓力觸發暫停",
            }],
        }], coding_by_id)
        self.assertEqual(directions, [])
        self.assertTrue(any("只有 1 個" in reason for reason in reasons))

    def test_research_direction_accepts_different_grounded_anchors(self):
        coding_by_id = {
            "C00001": {"code": "阻礙出現時暫停操作", "evidence_quote": "我會先停一下"},
            "C00002": {"code": "反覆受阻後放棄任務", "evidence_quote": "後來就不做了"},
        }
        directions, reasons = open_coding._validate_research_directions([{
            "direction": "從暫停到放棄的中斷歷程",
            "possible_finding": "參與者可能由短暫暫停逐步走向放棄任務。",
            "research_opportunity": "比較哪些阻礙只造成暫停，哪些會累積成放棄。",
            "shared_concept": "中斷歷程",
            "evidence_links": [
                {"candidate_id": "C00001", "connection": "提供歷程前段的暫停反應。",
                 "anchor_quote": "暫停操作"},
                {"candidate_id": "C00002", "connection": "提供歷程後段的放棄結果。",
                 "anchor_quote": "放棄任務"},
            ],
        }], coding_by_id)
        self.assertEqual(reasons, [])
        self.assertEqual(directions[0]["candidate_ids"], ["C00001", "C00002"])

    def test_focused_refinement_ceiling_keeps_room_beyond_minimum(self):
        codings = [{
            "segment_id": f"S{index:06d}", "evidence_quote": f"合成證據內容 {index}",
            "code": f"合成 code {index}", "research_relevance": 4,
            "behavior_pattern": 3, "evidence_strength": 4,
            "opportunity_potential": 2, "inference_risk": 1,
        } for index in range(1, 25)]
        args = Namespace(min_codes=10)
        with patch.object(open_coding, "_focused_select_resilient", return_value=codings[:12]) as select:
            selected = open_coding.refine_codings("合成研究規則", codings, args)
        self.assertEqual(select.call_args.args[3], 12)
        self.assertEqual(len(selected), 12)

    def test_focused_selection_rechecks_when_result_sticks_to_minimum(self):
        codings = [{
            "segment_id": f"S{index:06d}", "evidence_quote": f"合成證據內容 {index}",
            "code": f"合成 code {index}", "research_relevance": 4,
            "behavior_pattern": 3, "evidence_strength": 4,
            "opportunity_potential": 2, "inference_risk": 1,
        } for index in range(1, 21)]
        with patch.object(
            open_coding, "_focused_select",
            side_effect=[codings[:10], codings[10:12]],
        ) as select:
            result = open_coding._focused_select_resilient(
                "合成研究規則", codings, Namespace(), 15, "最終聚焦精選", 10,
            )
        self.assertEqual(select.call_count, 2)
        self.assertEqual(len(result), 12)
        self.assertEqual(select.call_args.kwargs["reference_selected"], codings[:10])

    def test_codings_are_read_in_transcript_order_not_score_order(self):
        segments = open_coding.segment_transcript("角色：第一句。\n角色：第二句。\n角色：第三句。")
        codings = [
            {"segment_id": "S000003", "analytic_score": 99},
            {"segment_id": "S000001", "analytic_score": 55},
            {"segment_id": "S000002", "analytic_score": 88},
        ]
        ordered = open_coding.sort_codings_in_transcript_order(codings, segments)
        self.assertEqual([row["segment_id"] for row in ordered], ["S000001", "S000002", "S000003"])

    def test_same_context_similar_codes_can_merge_with_grounded_anchors(self):
        segments = open_coding.segment_transcript("角色：先暫停操作。\n角色：後來仍然放棄任務。")
        codings = [
            {
                "segment_id": "S000001", "supporting_segment_ids": ["S000001", "S000002"],
                "evidence_quote": "先暫停操作", "evidence_context": "先暫停操作。\n後來仍然放棄任務。",
                "code": "遇到阻礙時先暫停操作", "rationale": "描述暫停反應。",
            },
            {
                "segment_id": "S000002", "supporting_segment_ids": ["S000001", "S000002"],
                "evidence_quote": "放棄任務", "evidence_context": "先暫停操作。\n後來仍然放棄任務。",
                "code": "遇到阻礙時由暫停走向放棄任務", "rationale": "描述後續結果。",
            },
        ]
        response = {"results": [{
            "action": "merge", "source_candidate_ids": ["C00001", "C00002"],
            "code": "遇到阻礙時先暫停操作，之後仍可能放棄任務。",
            "rationale": "兩筆語證描述同一段由暫停走向放棄的歷程。",
            "evidence_anchors": [
                {"candidate_id": "C00001", "anchor_quote": "暫停操作"},
                {"candidate_id": "C00002", "anchor_quote": "放棄任務"},
            ],
        }]}
        args = Namespace(host="local", model="test", timeout=10)
        with patch.object(open_coding, "ollama_chat", return_value=response):
            merged = open_coding.consolidate_context_codings(
                "合成研究規則", codings, segments, args,
            )
        self.assertEqual(len(merged), 1)
        self.assertIn("暫停操作", merged[0]["code"])
        self.assertEqual(merged[0]["supporting_segment_ids"], ["S000001", "S000002"])

    def test_same_context_opposing_codes_trigger_consistency_group(self):
        codings = [
            {"segment_id": "S000001", "supporting_segment_ids": ["S000001"], "code": "願意接受協助"},
            {"segment_id": "S000001", "supporting_segment_ids": ["S000001"], "code": "不願意接受協助"},
        ]
        groups = open_coding.context_consistency_groups(codings)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_research_direction_failure_does_not_rank_top_codes(self):
        text = open_coding.render_research_directions({
            "research_directions": [],
            "research_directions_note": "合成驗證未通過",
            "coding_by_id": {
                "C00001": {"code": "高分 code 甲", "analytic_score": 99},
                "C00002": {"code": "高分 code 乙", "analytic_score": 98},
            },
        })
        self.assertIn("沒有用最高分 codes 冒充研究方向", text)
        self.assertNotIn("高分 code 甲", text)
        self.assertNotIn("優先複核高分析分數", text)

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

    def test_invalid_new_primary_falls_back_to_existing_primary(self):
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
        self.assertEqual(rows[0]["segment_id"], original["segment_id"])
        self.assertEqual(rows[0]["evidence_quote"], original["evidence_quote"])

    def test_evidence_refinement_adds_topic_anchor_and_excludes_next_topic(self):
        segments = open_coding.segment_transcript("\n".join([
            "提問者：使用合成功能甲之後，你如何評價它？",
            "+ 合成功能甲讓我比較容易完成測試任務。",
            "那我們接下來問合成功能乙的使用情況。",
        ]), context_radius=12)
        primary = segments[1]
        original = {
            "segment_id": primary.id, "supporting_segment_ids": [primary.id],
            "evidence_context": primary.full_context, "evidence_quote": primary.text,
            "code": "體驗合成功能甲後，參與者認為它降低任務難度。",
            "rationale": "原句提供功能評價。", "full_context": primary.full_context,
        }
        result = {"contexts": [{
            "candidate_id": "C00001", "primary_segment_id": primary.id,
            "supporting_segment_ids": ["S000002", "S000003"],
            "supporting_segments": [], "reason": "合成模型範圍。",
        }]}
        args = Namespace(host="local", model="test", timeout=10)
        with patch.object(open_coding, "ollama_chat", return_value=result):
            rows = open_coding.refine_evidence_ranges("合成研究規則", [original], segments, args)
        self.assertEqual(rows[0]["supporting_segment_ids"], ["S000001", "S000002"])
        self.assertIn("合成功能甲之後", rows[0]["evidence_context"])
        self.assertNotIn("合成功能乙", rows[0]["evidence_context"])

    def test_interviewer_paraphrase_without_question_mark_is_rejected(self):
        segments = open_coding.segment_transcript("你覺得合成紀錄沒有幫助")
        result = {"codings": [{
            "segment_id": "S000001", "evidence_quote": "你覺得合成紀錄沒有幫助",
            "code": "錯把訪員重述當成參與者觀點", "rationale": "合成測試", "confidence": 0.9,
        }]}
        self.assertEqual(open_coding.validate_codings(result, segments), [])

    def test_minimum_context_floor_reaches_five_without_crossing_topic_switch(self):
        segments = open_coding.segment_transcript("\n".join([
            "提問者：請說明合成主題甲的完整體驗？",
            "參與者：先發生合成情境一。",
            "參與者：接著採取合成行動二。",
            "參與者：然後觀察合成結果三。",
            "參與者：最後形成一個完整評價。",
            "那我們接下來問合成主題乙。",
        ]), context_radius=12)
        ids = open_coding._ensure_minimum_context_ids(
            ["S000005"], "S000005", {item.id for item in segments}, segments, 5
        )
        self.assertEqual(ids, [f"S{i:06d}" for i in range(1, 6)])
        self.assertNotIn("S000006", ids)

    def test_context_floor_still_applies_when_refinement_times_out(self):
        segments = open_coding.segment_transcript("\n".join([
            "提問者：請說明合成主題甲的完整體驗？",
            "參與者：先發生合成情境一。",
            "參與者：接著採取合成行動二。",
            "參與者：然後觀察合成結果三。",
            "參與者：最後形成一個完整評價。",
        ]), context_radius=12)
        primary = segments[4]
        original = {
            "segment_id": primary.id, "supporting_segment_ids": [primary.id],
            "evidence_context": primary.text, "evidence_quote": primary.text,
            "code": "合成主題甲形成完整評價。", "rationale": "合成理由",
            "full_context": primary.full_context,
        }
        args = Namespace(host="local", model="test", timeout=1, min_context_segments=5)
        with patch.object(open_coding, "ollama_chat", side_effect=RuntimeError("模型請求逾時（1 秒）")):
            rows = open_coding.refine_evidence_ranges("合成研究規則", [original], segments, args)
        self.assertEqual(len(rows[0]["supporting_segment_ids"]), 5)

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

    def test_sparse_large_chunk_splits_and_preserves_parent_candidate(self):
        segments = open_coding.segment_transcript("\n".join(
            f"參與者：合成實質經驗片段{i}包含具體行動。" for i in range(1, 13)
        ))
        args = Namespace(host="local", model="test", timeout=10, retries=0, min_codes=0)
        calls = []

        def result_for(segment_id, evidence):
            return {"codings": [{
                "segment_id": segment_id, "evidence_quote": evidence,
                "code": f"由{segment_id}形成的合成行為模式。",
                "rationale": "合成來源直接支持。", "confidence": 0.9,
            }]}

        def fake_chat(_host, _model, _system, prompt, _timeout):
            calls.append(prompt)
            visible_ids = [
                item.id for item in segments
                if f'"segment_id": "{item.id}", "speaker"' in prompt
            ]
            if len(visible_ids) >= 12:
                return result_for("S000001", segments[0].text)
            if "S000001" in visible_ids:
                return result_for("S000002", segments[1].text)
            return result_for("S000010", segments[9].text)

        with patch.object(open_coding, "ollama_chat", side_effect=fake_chat):
            rows, skipped = open_coding.analyze_chunk("合成研究規則", segments, args, "測試區塊")
        self.assertEqual(skipped, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            {row["segment_id"] for row in rows},
            {"S000001", "S000002", "S000010"},
        )

    def test_parent_child_exact_candidate_is_deduplicated(self):
        row = {"segment_id": "S000001", "evidence_quote": "合成語證", "code": "合成 code"}
        self.assertEqual(open_coding._merge_unique_codings([row], [dict(row)]), [row])

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
            profile_result = {
                "participant": {"value": "參與者", "segment_id": "S000001",
                                "evidence_quote": "合成來源句"},
                "age": {"value": "未明", "segment_id": "", "evidence_quote": ""},
                "identity_facts": [], "characteristics": [],
                "interview_summary": "合成測試摘要。",
            }
            directions_result = {"research_directions": []}
            context_result = {"contexts": [{
                "candidate_id": "C00001", "primary_segment_id": "S000001",
                "supporting_segment_ids": ["S000001"], "reason": "單句已完整。",
            }]}
            with patch.object(
                open_coding, "ollama_chat",
                side_effect=[coding_result, context_result, profile_result, directions_result],
            ):
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
        self.assertEqual(
            open_coding_gui.status_context_for_log(
                "區塊 2 候選過少（1 個／8 個實質片段），自動拆成 7 句 + 8 句重掃"
            ),
            "初始 Coding 候選過少重掃",
        )


if __name__ == "__main__":
    unittest.main()
