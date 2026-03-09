"""Tests for VectorStore class."""

import sys
import os
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zoterorag.models import Sentence
from zoterorag.vector_store import VectorStore


class TestVectorStore:
    """LanceDB-backed VectorStore behavior tests."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def vector_store(self, temp_dir):
        return VectorStore(persist_directory=str(temp_dir))

    def _sentence(
        self,
        sentence_id: str,
        document_id: str,
        text: str,
        sentence_index: int,
        page: int = 1,
        page_section: int | None = 1,
    ) -> Sentence:
        return Sentence(
            id=sentence_id,
            document_id=document_id,
            page=page,
            page_section=page_section,
            sentence_index=sentence_index,
            text=text,
        )

    # --- Dimensions ---

    def test_detected_dimension_empty_store(self, vector_store):
        assert vector_store.get_detected_dimension() is None
        assert vector_store.has_dimension_mismatch(1024) is False

    def test_detected_dimension_after_reload(self, temp_dir):
        store = VectorStore(persist_directory=str(temp_dir))
        store.add_sentences(
            [self._sentence("doc1_sent_0", "doc1", "alpha", 0)],
            [[0.1, 0.2, 0.3, 0.4]],
            "doc1",
        )

        reloaded = VectorStore(persist_directory=str(temp_dir))
        assert reloaded.get_detected_dimension() == 4
        assert reloaded.has_dimension_mismatch(8) is True

    # --- Add/Get/Search ---

    def test_add_sentences_and_get_sentences(self, vector_store):
        sentences = [
            self._sentence("doc1_sent_0", "doc1", "A sentence [1].", 0),
            self._sentence("doc1_sent_1", "doc1", "Another sentence.", 1),
        ]
        sentences[0].citation_numbers = [1]
        sentences[0].referenced_texts = ["A. Author. Title. 2020."]
        sentences[0].referenced_bibtex = ["@article{author2020,title={Title}}"]

        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        vector_store.add_sentences(sentences, embeddings, "doc1")

        out = vector_store.get_sentences("doc1")
        assert len(out) == 2
        assert [s.sentence_index for s in out] == [0, 1]
        assert out[0].citation_numbers == [1]
        assert out[0].referenced_texts
        assert out[0].referenced_bibtex

    def test_search_sentence_ids_cosine_scores(self, vector_store):
        vector_store.add_sentences(
            [
                self._sentence("a", "doc1", "alpha", 0),
                self._sentence("b", "doc1", "beta", 1),
                self._sentence("c", "doc2", "gamma", 0),
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            "doc1",
        )

        # Add doc2 separately so document_key is persisted correctly.
        vector_store.add_sentences(
            [self._sentence("c", "doc2", "gamma", 0)],
            [[1.0, 0.0, 0.0, 0.0]],
            "doc2",
        )

        ids, scores, metas = vector_store.search_sentence_ids(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            document_key="doc1",
            top_k=2,
        )

        assert ids
        assert len(ids) == len(scores) == len(metas)
        assert ids[0] == "a"
        assert all(-1.0 <= s <= 1.0 for s in scores)
        assert metas[0]["document_key"] == "doc1"

    def test_get_sentence_texts_by_ids(self, vector_store):
        vector_store.add_sentences(
            [
                self._sentence("x", "doc1", "X text", 0),
                self._sentence("y", "doc1", "Y text", 1),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
            "doc1",
        )

        out = vector_store.get_sentence_texts_by_ids(["x", "y", "z"])
        assert out["x"] == "X text"
        assert out["y"] == "Y text"
        assert "z" not in out

    def test_get_sentence_metadatas_by_ids(self, vector_store):
        s = self._sentence("m1", "doc1", "meta", 3, page=2, page_section=None)
        s.citation_numbers = [2, 5]
        vector_store.add_sentences([s], [[0.3, 0.7]], "doc1")

        out = vector_store.get_sentence_metadatas_by_ids(["m1"])
        assert out["m1"]["document_key"] == "doc1"
        assert out["m1"]["page"] == 2
        assert out["m1"]["sentence_index"] == 3
        assert out["m1"]["citation_numbers"] == [2, 5]

    # --- Embedded doc helpers + cleanup ---

    def test_is_document_embedded(self, vector_store):
        vector_store.add_sentences(
            [self._sentence("d1_s0", "doc1", "text", 0)],
            [[1.0, 0.0]],
            "doc1",
        )
        assert vector_store.is_document_embedded("doc1") is True
        assert vector_store.is_document_embedded("missing") is False

    def test_delete_document(self, vector_store):
        vector_store.add_sentences(
            [self._sentence("d1_s0", "doc1", "text", 0)],
            [[1.0, 0.0]],
            "doc1",
        )
        assert vector_store.get_sentence_count() == 1

        vector_store.delete_document("doc1")
        assert vector_store.get_sentence_count() == 0

    def test_clear_all(self, vector_store):
        vector_store.add_sentences(
            [self._sentence("d1_s0", "doc1", "text", 0)],
            [[1.0, 0.0]],
            "doc1",
        )
        vector_store.clear_all()

        assert vector_store.get_sentence_count() == 0
        assert vector_store.get_detected_dimension() is None

    def test_embedded_documents_json_roundtrip(self, vector_store):
        vector_store.save_embedded_documents({"doc1": 12, "doc2": 9})
        assert vector_store.get_embedded_documents() == {"doc1": 12, "doc2": 9}

        vector_store.update_embedded_document("doc3", 4)
        assert vector_store.get_embedded_documents()["doc3"] == 4

    def test_get_document_title_compatibility(self, vector_store):
        assert vector_store.get_document_title("doc1") is None

    def test_index_type_defaults_to_hnsw(self, temp_dir, monkeypatch):
        monkeypatch.delenv("LANCEDB_INDEX_TYPE", raising=False)
        store = VectorStore(persist_directory=str(temp_dir))
        assert store._index_type == "IVF_HNSW_SQ"

    def test_index_type_reads_env_case_insensitive(self, temp_dir, monkeypatch):
        monkeypatch.setenv("LANCEDB_INDEX_TYPE", "ivf_flat")
        store = VectorStore(persist_directory=str(temp_dir))
        assert store._index_type == "IVF_FLAT"

    def test_invalid_index_type_falls_back_to_ivf_flat(self, temp_dir, monkeypatch):
        monkeypatch.setenv("LANCEDB_INDEX_TYPE", "banana")
        store = VectorStore(persist_directory=str(temp_dir))
        assert store._index_type == "IVF_FLAT"

    def test_add_sentences_handles_empty_first_batch_schema(self, vector_store):
        first = self._sentence(
            "doc1_sent_0", "doc1", "First sentence.", 0, page_section=None
        )
        second = self._sentence("doc1_sent_1", "doc1", "Second sentence [4].", 1)
        second.citation_numbers = [4]
        second.referenced_texts = ["A. Author. Paper. 2020."]
        second.referenced_bibtex = ["@article{author2020,title={Paper}}"]

        vector_store.add_sentences([first], [[0.1, 0.2, 0.3]], "doc1")
        vector_store.add_sentences([second], [[0.3, 0.2, 0.1]], "doc1")

        out = vector_store.get_sentences("doc1")
        assert len(out) == 2
        assert out[0].page_section is None
        assert out[1].page_section == 1
        assert out[1].citation_numbers == [4]
        assert out[1].referenced_texts == ["A. Author. Paper. 2020."]
        assert out[1].referenced_bibtex == ["@article{author2020,title={Paper}}"]

    def test_repairs_legacy_null_typed_schema_on_reload(self, temp_dir):
        db_dir = temp_dir / "lancedb"
        db_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_dir))
        legacy = pa.table(
            {
                "id": pa.array(["legacy-1"]),
                "vector": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32(), 3)),
                "document": pa.array(["legacy text"]),
                "document_key": pa.array(["doc1"]),
                "page": pa.array([1], type=pa.int64()),
                "page_section": pa.nulls(1),
                "sentence_index": pa.array([0], type=pa.int64()),
                "citation_numbers": pa.array([[]]),
                "referenced_texts": pa.array([[]]),
                "referenced_bibtex": pa.array([[]]),
            }
        )
        db.create_table("sentences", data=legacy, mode="overwrite")

        store = VectorStore(persist_directory=str(temp_dir))
        repaired_schema = store.sentences_table.schema
        assert repaired_schema.field("page_section").type == pa.int64()
        assert repaired_schema.field("referenced_texts").type == pa.list_(pa.string())
        assert repaired_schema.field("referenced_bibtex").type == pa.list_(pa.string())

        follow_up = self._sentence("doc1_sent_1", "doc1", "new text", 1, page_section=2)
        follow_up.referenced_texts = ["Ref text"]
        follow_up.referenced_bibtex = ["@article{new}"]
        store.add_sentences([follow_up], [[0.4, 0.5, 0.6]], "doc1")

        out = store.get_sentences("doc1")
        assert len(out) == 2
        assert out[1].page_section == 2
        assert out[1].referenced_texts == ["Ref text"]
        assert out[1].referenced_bibtex == ["@article{new}"]

    def test_reader_refreshes_after_external_write_without_restart(self, temp_dir):
        reader = VectorStore(persist_directory=str(temp_dir))
        writer = VectorStore(persist_directory=str(temp_dir))

        writer.add_sentences(
            [self._sentence("late_doc_sent_0", "late_doc", "Fresh text", 0)],
            [[1.0, 0.0, 0.0]],
            "late_doc",
        )

        assert reader.get_sentence_count() == 1
        assert reader.is_document_embedded("late_doc") is True

        ids, scores, metas = reader.search_sentence_ids(
            query_embedding=[1.0, 0.0, 0.0],
            document_key="late_doc",
            top_k=1,
        )
        assert ids == ["late_doc_sent_0"]
        assert len(scores) == 1
        assert metas[0]["document_key"] == "late_doc"
        assert reader.get_sentence_texts_by_ids(ids) == {
            "late_doc_sent_0": "Fresh text"
        }

    def test_add_sentences_skips_delete_for_new_document(
        self, vector_store, monkeypatch: pytest.MonkeyPatch
    ):
        vector_store.add_sentences(
            [self._sentence("doc1_sent_0", "doc1", "alpha", 0)],
            [[1.0, 0.0, 0.0]],
            "doc1",
        )

        delete_calls: list[str] = []
        original_delete = vector_store.sentences_table.delete

        def tracking_delete(where: str):
            delete_calls.append(where)
            return original_delete(where)

        monkeypatch.setattr(vector_store.sentences_table, "delete", tracking_delete)

        vector_store.add_sentences(
            [
                self._sentence("doc2_sent_0", "doc2", "beta", 0),
                self._sentence("doc2_sent_1", "doc2", "gamma", 1),
            ],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "doc2",
            batch_size=1,
        )

        assert delete_calls == []
        assert vector_store.get_sentence_count() == 3

    def test_delete_document_removes_embedded_document_metadata(self, vector_store):
        vector_store.add_sentences(
            [self._sentence("doc1_sent_0", "doc1", "alpha", 0)],
            [[1.0, 0.0, 0.0]],
            "doc1",
        )
        vector_store.update_embedded_document("doc1", 1)

        assert vector_store.get_embedded_documents()["doc1"] == 1

        vector_store.delete_document("doc1")

        assert "doc1" not in vector_store.get_embedded_documents()
        assert vector_store.get_sentence_count() == 0
