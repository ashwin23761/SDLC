"""Tests for MongoDB run history persistence and crew_runner integration."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet
from pymongo import DESCENDING

from src import crew_runner, db


class _FakeCursor:
    def __init__(self, docs: list[dict], projection: dict | None = None):
        self._docs = docs
        self._projection = projection

    def sort(self, key_spec, direction=None):
        if isinstance(key_spec, str):
            specs = [(key_spec, direction or 1)]
        else:
            specs = list(key_spec)
        docs = list(self._docs)
        for field, order in reversed(specs):
            docs.sort(key=lambda d: d.get(field, ""), reverse=(order == DESCENDING or order == -1))
        self._docs = docs
        return self

    def limit(self, count: int):
        self._docs = self._docs[:count]
        return self

    def __iter__(self):
        for doc in self._docs:
            yield self._apply_projection(doc)

    def _apply_projection(self, doc: dict) -> dict:
        if not self._projection:
            return copy.deepcopy(doc)
        include_keys = {k for k, v in self._projection.items() if v}
        result = {}
        for key in include_keys:
            if "." in key:
                parts = key.split(".")
                cur_src = doc
                cur_dst = result
                ok = True
                for part in parts[:-1]:
                    if isinstance(cur_src, dict) and part in cur_src:
                        cur_src = cur_src[part]
                        cur_dst = cur_dst.setdefault(part, {})
                    else:
                        ok = False
                        break
                if ok and isinstance(cur_src, dict) and parts[-1] in cur_src:
                    cur_dst[parts[-1]] = copy.deepcopy(cur_src[parts[-1]])
            elif key in doc:
                result[key] = copy.deepcopy(doc[key])
        return result


class _FakeCollection:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.indexes: list = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def replace_one(self, filter_query: dict, replacement: dict, upsert: bool = False):
        run_id = filter_query["run_id"]
        if run_id in self.docs or upsert:
            stored = copy.deepcopy(replacement)
            stored["_id"] = run_id
            self.docs[run_id] = stored

    def find_one(self, filter_query: dict):
        run_id = filter_query.get("run_id")
        doc = self.docs.get(run_id)
        return copy.deepcopy(doc) if doc is not None else None

    def find(self, query: dict | None = None, projection: dict | None = None):
        query = query or {}
        matched = []
        for doc in self.docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                matched.append(copy.deepcopy(doc))
        return _FakeCursor(matched, projection=projection)

    def delete_one(self, filter_query: dict):
        run_id = filter_query.get("run_id")
        deleted = 1 if run_id in self.docs else 0
        self.docs.pop(run_id, None)
        return type("DeleteResult", (), {"deleted_count": deleted})()

    def count_documents(self, query: dict):
        return sum(1 for doc in self.docs.values() if all(doc.get(k) == v for k, v in query.items()))


def _sample_record(run_id: str = "20261128-010203-abc123", model: str = "llama3.2:3b", scenario: str = "Retail loan approval workflow") -> dict:
    return {
        "run_id": run_id,
        "created_by": "analyst1",
        "model_name": model,
        "duration_seconds": 12.5,
        "scenario_redacted": scenario,
        "reference_documents": ["loan_approval_policy.md"],
        "uploaded_documents": [],
        "retrieved_chunks": [],
        "requirements": {
            "summary": "Loan approval summary.",
            "functional_requirements": [
                {
                    "id": "FR-01",
                    "category": "Decisioning",
                    "description": "The system shall route loans above 50,000 to an underwriter.",
                    "priority": "High",
                    "reasoning": "Loan policy s.3.1",
                }
            ],
            "non_functional_requirements": [
                {
                    "id": "NFR-01",
                    "category": "Security",
                    "description": "The system shall encrypt customer data at rest.",
                    "priority": "High",
                    "reasoning": "Loan policy s.5.1",
                }
            ],
            "assumptions": ["Credit bureau API is available."],
        },
        "sdlc": {
            "project_characteristics": ["High regulatory criticality"],
            "ranked_recommendations": [
                {"approach": "Agile-V-Model hybrid", "confidence_percent": 82, "justification": "Balances compliance and iteration."}
            ],
            "recommended_approach": "Agile-V-Model hybrid",
            "workflow_notes": "Include compliance gates.",
            "approval_note": "Requires sign-off.",
        },
        "extended_analysis": {
            "human_approval_gates": [{"item": "FR-01", "required_approver": "Compliance", "status": "Pending"}],
        },
    }


class MongoHistoryTests(unittest.TestCase):
    def setUp(self):
        self.col = _FakeCollection()

    def test_save_get_list_delete_plaintext(self):
        with patch.dict(os.environ, {"FINANCE_REQ_AUDIT_ENCRYPTION_KEY": ""}, clear=False):
            r1 = _sample_record("20261127-100000-111111", model="llama3.2:3b", scenario="KYC onboarding flow")
            r2 = _sample_record("20261128-100000-222222", model="qwen2.5:7b", scenario="Wire transfer fraud detection")
            db.save_run_record(r1, collection=self.col)
            db.save_run_record(r2, collection=self.col)

            self.assertEqual(db.count_run_records(collection=self.col), 2)
            loaded = db.get_run_record("20261128-100000-222222", collection=self.col)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["scenario_redacted"], "Wire transfer fraud detection")
            self.assertNotIn("_id", loaded)

            summaries = db.list_run_summaries(collection=self.col)
            self.assertEqual([s["run_id"] for s in summaries], ["20261128-100000-222222", "20261127-100000-111111"])

            filtered = db.list_run_summaries(search_query="fraud", collection=self.col)
            self.assertEqual([s["run_id"] for s in filtered], ["20261128-100000-222222"])

            model_filtered = db.list_run_summaries(model_filter="llama3.2:3b", collection=self.col)
            self.assertEqual([s["run_id"] for s in model_filtered], ["20261127-100000-111111"])

            self.assertTrue(db.delete_run_record("20261127-100000-111111", collection=self.col))
            self.assertIsNone(db.get_run_record("20261127-100000-111111", collection=self.col))

    def test_save_and_read_encrypted(self):
        key = Fernet.generate_key().decode("ascii")
        with patch.dict(os.environ, {"FINANCE_REQ_AUDIT_ENCRYPTION_KEY": key}, clear=False):
            rec = _sample_record("20261128-120000-enc123", scenario="Confidential sanctions screening")
            db.save_run_record(rec, collection=self.col)

            raw_doc = self.col.docs["20261128-120000-enc123"]
            self.assertTrue(raw_doc["encrypted"])
            self.assertNotIn("requirements", raw_doc)
            self.assertNotIn("scenario_redacted", raw_doc)

            decoded = db.get_run_record("20261128-120000-enc123", collection=self.col)
            self.assertEqual(decoded["scenario_redacted"], "Confidential sanctions screening")
            self.assertEqual(decoded["requirements"]["functional_requirements"][0]["id"], "FR-01")

            summaries = db.list_run_summaries(search_query="sanctions", collection=self.col)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["run_id"], "20261128-120000-enc123")

    def test_sync_existing_outputs_and_reconstruct_legacy_runs(self):
        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        with patch.dict(os.environ, {"FINANCE_REQ_AUDIT_ENCRYPTION_KEY": ""}, clear=False):
            stats = db.sync_outputs_dir_to_mongo(str(outputs_dir), collection=self.col)
            self.assertGreaterEqual(stats["synced"], 5)
            self.assertEqual(stats["errors"], [])

            # Re-sync should skip all already-present runs.
            stats_second = db.sync_outputs_dir_to_mongo(str(outputs_dir), collection=self.col)
            self.assertEqual(stats_second["synced"], 0)
            self.assertEqual(stats_second["skipped"], stats["synced"])

            # Verify a legacy record (without extended_analysis) reconstructs into RunResult cleanly.
            legacy_doc = db.get_run_record("20260927-202944", collection=self.col)
            self.assertIsNotNone(legacy_doc)
            legacy_result = crew_runner.record_to_run_result(legacy_doc)
            self.assertEqual(legacy_result.run_id, "20260927-202944")
            self.assertIn("human_approval_gates", legacy_result.extended_analysis)
            self.assertGreater(len(legacy_result.requirements.functional_requirements), 0)

            # Verify a modern record reconstructs into RunResult cleanly.
            modern_doc = db.get_run_record("20261126-165306-128070", collection=self.col)
            self.assertIsNotNone(modern_doc)
            modern_result = crew_runner.record_to_run_result(modern_doc)
            self.assertEqual(modern_result.run_id, "20261126-165306-128070")

    def test_crew_runner_mongo_roundtrip_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(crew_runner, "OUTPUTS_DIR", tmpdir), patch.dict(
                os.environ, {"FINANCE_REQ_AUDIT_ENCRYPTION_KEY": ""}, clear=False
            ):
                rec = _sample_record("20261128-150000-round1")
                Path(tmpdir, "run_20261128-150000-round1.json").write_text(json.dumps(rec), encoding="utf-8")

                with (
                    patch("src.crew_runner.get_run_record", side_effect=lambda rid: db.get_run_record(rid, collection=self.col)),
                    patch("src.crew_runner.save_run_record", side_effect=lambda r: db.save_run_record(r, collection=self.col)),
                    patch("src.crew_runner.check_mongo_connection", return_value=(True, "test.runs")),
                    patch("src.crew_runner.sync_local_history_to_mongo", side_effect=lambda: db.sync_outputs_dir_to_mongo(tmpdir, collection=self.col)),
                    patch("src.crew_runner.list_run_summaries", side_effect=lambda **kw: db.list_run_summaries(collection=self.col, **kw)),
                ):
                    items, backend, sync_stats = crew_runner.list_run_history()
                    self.assertEqual(backend, "mongodb")
                    self.assertEqual(sync_stats["synced"], 1)
                    self.assertEqual(len(items), 1)

                    digest = crew_runner.save_approval_changes(
                        "20261128-150000-round1",
                        [{"item": "FR-01", "required_approver": "Compliance", "status": "Approved", "reviewer_notes": "OK", "modified_output": ""}],
                        actor="approver1",
                    )
                    self.assertEqual(len(digest), 64)
                    updated_mongo = db.get_run_record("20261128-150000-round1", collection=self.col)
                    self.assertEqual(updated_mongo["extended_analysis"]["human_approval_gates"][0]["status"], "Approved")

                    fb_digest = crew_runner.save_evaluation_feedback(
                        "20261128-150000-round1",
                        {"satisfaction_1_to_5": 5, "reviewer_notes": "Accurate"},
                        actor="approver1",
                    )
                    self.assertEqual(len(fb_digest), 64)

                # Fallback when MongoDB is unreachable
                with (
                    patch("src.crew_runner.check_mongo_connection", return_value=(False, "unreachable")),
                    patch("src.crew_runner.get_run_record", return_value=None),
                ):
                    items, backend, _ = crew_runner.list_run_history()
                    self.assertEqual(backend, "local_files")
                    self.assertEqual(len(items), 1)
                    self.assertEqual(items[0]["run_id"], "20261128-150000-round1")

    def test_app_ui_contract_and_history_helpers(self):
        import py_compile

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        py_compile.compile(str(app_path), doraise=True)

        cfg = db.get_mongo_config()
        self.assertIn("uri", cfg)
        self.assertIn("db_name", cfg)
        self.assertIn("collection_name", cfg)

        with patch.dict(os.environ, {"FINANCE_REQ_MONGODB_ENABLED": "0"}, clear=False):
            ok, msg = db.check_mongo_connection()
            self.assertFalse(ok)
            self.assertIn("disabled", msg.lower())

        with patch.dict(os.environ, {"FINANCE_REQ_AUDIT_ENCRYPTION_KEY": ""}, clear=False):
            db.sync_outputs_dir_to_mongo(str(Path(__file__).resolve().parent.parent / "outputs"), collection=self.col)
            with (
                patch("src.crew_runner.get_run_record", side_effect=lambda rid: db.get_run_record(rid, collection=self.col)),
                patch("src.crew_runner.check_mongo_connection", return_value=(True, "finance_req_agents.runs")),
                patch("src.crew_runner.sync_local_history_to_mongo", return_value={"synced": 0, "skipped": 5, "errors": []}),
                patch("src.crew_runner.list_run_summaries", side_effect=lambda **kw: db.list_run_summaries(collection=self.col, **kw)),
            ):
                summaries, backend, _ = crew_runner.list_run_history(limit=10)
                self.assertEqual(backend, "mongodb")
                self.assertGreaterEqual(len(summaries), 5)
                loaded = crew_runner.load_run_result(summaries[0]["run_id"])
                self.assertEqual(loaded.run_id, summaries[0]["run_id"])
                self.assertTrue(loaded.storage_info.get("loaded_from_history"))


if __name__ == "__main__":
    unittest.main()
