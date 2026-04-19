from __future__ import annotations

import json
from pathlib import Path
import unittest

from utils.parser import parse_cisi_file, parse_cisi_text, parse_cisi_rel_file


class TestParser(unittest.TestCase):
    def test_parse_cisi_text_handles_missing_tags(self) -> None:
        raw = (
            ".I 1\n.T\nDoc one\n.A\nAuthor one\n.W\nBody one\n"
            ".I 2\n.T\nDoc two\n.W\nBody two only\n"
        )
        docs = parse_cisi_text(raw)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["doc_id"], 1)
        self.assertEqual(docs[0]["title"], "Doc one")
        self.assertEqual(docs[0]["author"], "Author one")
        self.assertEqual(docs[0]["text"], "Body one")
        self.assertEqual(docs[1]["doc_id"], 2)
        self.assertEqual(docs[1]["title"], "Doc two")
        self.assertEqual(docs[1]["author"], "")
        self.assertEqual(docs[1]["text"], "Body two only")

    def test_parse_cisi_all_schema_and_count(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        all_path = project_root / "CISI_data" / "CISI.ALL"
        docs = parse_cisi_file(all_path)

        self.assertEqual(len(docs), 1460)
        required_keys = {"doc_id", "title", "author", "text", "raw_tags"}
        self.assertTrue(all(required_keys == set(d.keys()) for d in docs))

    def test_parse_cisi_qry_to_queries_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        qry_path = project_root / "CISI_data" / "CISI.QRY"
        parsed_queries = parse_cisi_file(qry_path)
        queries = [{"query_id": q["doc_id"], "text": q["text"]} for q in parsed_queries]

        self.assertEqual(len(queries), 112)
        self.assertTrue(all(set(q.keys()) == {"query_id", "text"} for q in queries))

    def test_parse_cisi_rel_qrels(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rel_path = project_root / "CISI_data" / "CISI.REL"
        qrels = parse_cisi_rel_file(rel_path)

        self.assertGreater(len(qrels), 0)
        self.assertIn(1, qrels)
        self.assertIsInstance(qrels[1], list)
        self.assertTrue(all(isinstance(doc_id, int) for doc_id in qrels[1]))

    def test_existing_exported_json_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        corpus_path = project_root / "data" / "corpus.json"
        queries_path = project_root / "data" / "queries.json"

        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        queries = json.loads(queries_path.read_text(encoding="utf-8"))

        self.assertEqual(len(corpus), 1460)
        self.assertEqual(len(queries), 112)
        self.assertTrue(all(set(d.keys()) == {"doc_id", "title", "author", "text"} for d in corpus))
        self.assertTrue(all(set(q.keys()) == {"query_id", "text"} for q in queries))


if __name__ == "__main__":
    unittest.main()
