"""Tests for query decomposition and retrieval-selection improvements.

These tests avoid loading local LLM, embedding, or cross-encoder models.  They
exercise the deterministic glue code added around query cleanup, prompt wording,
retrieval rescoring, retrieval-run merging, and generator-context selection.
"""

import sys
import types
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit


try:
    import llama_cpp  # noqa: F401
except ModuleNotFoundError:
    llama_cpp_stub = types.ModuleType("llama_cpp")

    class _UnavailableLlama:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("llama_cpp is not available in this test environment")

    class _UnavailableCache:
        pass

    llama_cpp_stub.Llama = _UnavailableLlama
    llama_cpp_stub.LlamaRAMCache = _UnavailableCache
    sys.modules["llama_cpp"] = llama_cpp_stub


class TestSubqueryCleanup:
    """Unit tests for cleaning model-generated query text."""

    @patch("src.query_enhancement.run_llama_cpp")
    def test_decompose_strips_numbering_labels_and_inverted_punctuation(self, mock_llm):
        """Subquery parsing removes common LLM formatting artifacts."""
        from src.query_enhancement import decompose_complex_query

        mock_llm.return_value = {
            "choices": [
                {
                    "text": (
                        "1) ¿What snapshot is read?\n"
                        "Output: What does first-committer-wins mean?\n"
                        "- Â¿Why can anomalies still occur?\n"
                        "4. What extra question should be ignored?"
                    )
                }
            ]
        }

        result = decompose_complex_query(
            "Under snapshot isolation, what snapshot is read, what does "
            "first-committer-wins mean, and why can anomalies still occur?",
            model_path="mock-model.gguf",
            max_sub_questions=3,
        )

        assert result == [
            "What snapshot is read?",
            "What does first-committer-wins mean?",
            "Why can anomalies still occur?",
        ]

    @patch("src.query_enhancement.run_llama_cpp")
    def test_decomposition_prompt_requires_standalone_context(self, mock_llm):
        """The prompt explicitly asks the model to preserve qualifiers."""
        from src.query_enhancement import decompose_complex_query

        mock_llm.return_value = {
            "choices": [
                {
                    "text": (
                        "What snapshot does a transaction read under snapshot isolation?"
                    )
                }
            ]
        }

        decompose_complex_query(
            "Under snapshot isolation, what snapshot does a transaction read?",
            model_path="mock-model.gguf",
        )

        prompt = mock_llm.call_args.args[0]
        assert "Preserve necessary topic qualifiers" in prompt
        assert "Do not make vague subquestions" in prompt

    def test_clean_generated_query_removes_echoes_and_uses_fallback(self):
        """Standalone-query cleanup removes labels and rejects huge echoes."""
        from src.query_enhancement import clean_generated_query

        assert (
            clean_generated_query(
                "Chat History: ... Output: ¿Why is BCNF useful?",
                fallback="Why is it useful?",
            )
            == "Why is BCNF useful?"
        )

        long_echo = "Output: " + " ".join(["unrelated"] * 100)
        assert clean_generated_query(long_echo, fallback="What is SQL?") == "What is SQL?"

    @patch("src.query_enhancement.run_llama_cpp")
    def test_contextualize_query_applies_generated_query_cleanup(self, mock_llm):
        """Contextualization returns the cleaned standalone question."""
        from src.query_enhancement import contextualize_query

        mock_llm.return_value = {
            "choices": [{"text": "Output: ¿Why is BCNF useful?"}]
        }

        result = contextualize_query(
            "Why is it useful?",
            history=[
                {"role": "user", "content": "What is BCNF?"},
                {"role": "assistant", "content": "BCNF is a normal form."},
            ],
            model_path="mock-model.gguf",
        )

        assert result == "Why is BCNF useful?"


class TestGroundedGenerationPrompts:
    """Tests for prompt instructions that reduce unsupported answer details."""

    def test_tutor_prompt_requires_answering_all_parts_from_excerpts(self):
        from src.generator import get_system_prompt

        prompt = get_system_prompt("tutor")

        assert "identify all parts" in prompt
        assert "Refer ONLY to the provided textbook excerpts" in prompt
        assert "Do not invent examples" in prompt
        assert "leave it out rather than guessing" in prompt

    def test_concise_and_detailed_prompts_discourage_invented_examples(self):
        from src.generator import get_system_prompt

        assert "Do not invent examples" in get_system_prompt("concise")
        assert "Do not invent examples" in get_system_prompt("detailed")

    def test_format_prompt_places_context_before_question(self):
        from src.generator import ANSWER_START, format_prompt

        prompt = format_prompt(
            ["Chunk A explains alpha.", "Chunk B explains beta."],
            "Explain alpha and beta.",
            system_prompt_mode="tutor",
        )

        assert prompt.index("Textbook Excerpts:") < prompt.index("Question:")
        assert "Chunk A explains alpha." in prompt
        assert "Explain alpha and beta." in prompt
        assert ANSWER_START in prompt


class TestRetrievalSelectionScoring:
    """Unit tests for deterministic retrieval/ranking helpers."""

    def test_definition_rerank_prefers_definition_over_incidental_mention(self):
        """A true definition should beat a chunk that merely mentions the term."""
        from src.retrieval_selection import rerank_with_query_overlap

        chunks = [
            (
                "SQL includes a references privilege that permits users to "
                "declare foreign keys when creating relations."
            ),
            (
                "A foreign-key constraint states that values in one relation "
                "must appear as primary-key values in another relation. "
                "Attribute set A is called a foreign key from the referencing "
                "relation."
            ),
        ]

        ordered, _ = rerank_with_query_overlap(
            "What is a foreign key?",
            [0, 1],
            [1.0, 0.9],
            chunks,
        )

        assert ordered == [1, 0]

    def test_non_definition_rerank_uses_query_term_overlap(self):
        """For broader questions, chunks matching more content terms move up."""
        from src.retrieval_selection import rerank_with_query_overlap

        chunks = [
            "This chunk discusses unrelated storage performance.",
            (
                "Snapshot isolation can allow anomalies because concurrent "
                "transactions may read old versions and miss read-write conflicts."
            ),
        ]

        ordered, _ = rerank_with_query_overlap(
            "Explain snapshot isolation anomalies and read-write conflicts.",
            [0, 1],
            [1.0, 0.9],
            chunks,
        )

        assert ordered == [1, 0]

    def test_merge_retrieval_runs_covers_each_subquery_before_global_fill(self):
        """Merged retrieval keeps one strong candidate per decomposed subquery."""
        from src.retrieval_selection import merge_retrieval_runs

        chunks = [
            "Global overview.",
            "Global backup.",
            "Alpha is defined here.",
            "Beta is defined here.",
        ]
        retrieval_runs = [
            {"question": "Explain alpha and beta.", "topk_idxs": [0, 1, 2, 3], "scores": [0.99, 0.98, 0.2, 0.1]},
            {"question": "What is alpha?", "topk_idxs": [2, 0], "scores": [0.95, 0.1]},
            {"question": "What is beta?", "topk_idxs": [3, 1], "scores": [0.95, 0.1]},
        ]

        selected, _ = merge_retrieval_runs(retrieval_runs, chunks, limit=3)

        assert selected == [2, 3, 0]

    def test_merge_retrieval_runs_deduplicates_shared_candidates(self):
        """A chunk selected for one subquery is not duplicated for another."""
        from src.retrieval_selection import merge_retrieval_runs

        chunks = [
            "Global overview.",
            "Shared alpha beta explanation.",
            "Beta-only explanation.",
        ]
        retrieval_runs = [
            {"question": "Explain alpha and beta.", "topk_idxs": [0, 1, 2], "scores": [0.99, 0.7, 0.6]},
            {"question": "What is alpha?", "topk_idxs": [1], "scores": [0.95]},
            {"question": "What is beta?", "topk_idxs": [1, 2], "scores": [0.96, 0.8]},
        ]

        selected, _ = merge_retrieval_runs(retrieval_runs, chunks, limit=3)

        assert selected == [1, 2, 0]
        assert len(selected) == len(set(selected))


class TestGeneratorContextSelection:
    """Tests for choosing the final chunks sent to the generator."""

    def test_generator_selection_preserves_subquery_coverage(self):
        from src.retrieval_selection import rerank_chunks_with_ids

        chunks = [
            "General overview chunk.",
            "Another global chunk.",
            "Background chunk.",
            "Alpha is a specific definition.",
            "Beta is another specific definition.",
        ]
        retrieval_runs = [
            {"question": "Explain alpha and beta.", "topk_idxs": [0, 1, 2, 3, 4], "scores": [0.9, 0.8, 0.7, 0.6, 0.5]},
            {"question": "What is alpha?", "topk_idxs": [3, 0], "scores": [0.95, 0.1]},
            {"question": "What is beta?", "topk_idxs": [4, 1], "scores": [0.95, 0.1]},
        ]

        _, sent_chunks = rerank_chunks_with_ids(
            "Explain alpha and beta.",
            [0, 1, 2, 3, 4],
            chunks,
            mode="",
            top_n=3,
            retrieval_runs=retrieval_runs,
        )

        assert [chunk["idx"] for chunk in sent_chunks] == [3, 4, 0]
        assert [chunk["selection_reason"] for chunk in sent_chunks] == [
            "subquery_coverage",
            "subquery_coverage",
            "global_rerank",
        ]

    def test_generator_selection_reuses_one_chunk_for_multiple_subqueries(self):
        from src.retrieval_selection import rerank_chunks_with_ids

        chunks = [
            "Alpha and beta are both explained here.",
            "Weak beta-only chunk.",
            "Global filler chunk.",
        ]
        retrieval_runs = [
            {"question": "Explain alpha and beta.", "topk_idxs": [0, 1, 2], "scores": [0.9, 0.8, 0.7]},
            {"question": "What is alpha?", "topk_idxs": [0], "scores": [0.95]},
            {"question": "What is beta?", "topk_idxs": [0, 1], "scores": [0.95, 0.5]},
        ]

        _, sent_chunks = rerank_chunks_with_ids(
            "Explain alpha and beta.",
            [0, 1, 2],
            chunks,
            mode="",
            top_n=2,
            retrieval_runs=retrieval_runs,
        )

        assert [chunk["idx"] for chunk in sent_chunks] == [0, 1]
        assert sent_chunks[0]["covered_queries"] == ["What is alpha?", "What is beta?"]

    def test_generator_selection_does_not_force_low_rerank_coverage(self):
        from src.retrieval_selection import rerank_chunks_with_ids

        chunks = [f"Global chunk {i}" for i in range(7)] + ["Weak forced coverage chunk"]
        retrieval_runs = [
            {"question": "Explain alpha and beta.", "topk_idxs": list(range(8)), "scores": [1.0] * 8},
            {"question": "What is beta?", "topk_idxs": [7], "scores": [0.95]},
        ]

        _, sent_chunks = rerank_chunks_with_ids(
            "Explain alpha and beta.",
            list(range(8)),
            chunks,
            mode="",
            top_n=3,
            retrieval_runs=retrieval_runs,
        )

        assert [chunk["idx"] for chunk in sent_chunks] == [0, 1, 2]
        assert all(chunk["selection_reason"] == "global_rerank" for chunk in sent_chunks)

    def test_duplicate_chunk_text_still_maps_to_distinct_chunk_ids(self):
        """Repeated chunk text should not collapse different retrieval ids."""
        from src.retrieval_selection import rerank_chunks_with_ids

        chunks = ["Duplicate text.", "Duplicate text.", "Different text."]

        _, sent_chunks = rerank_chunks_with_ids(
            "Explain duplicate chunks.",
            [0, 1, 2],
            chunks,
            mode="",
            top_n=3,
        )

        assert [chunk["idx"] for chunk in sent_chunks] == [0, 1, 2]


class TestMultiPartRetrievalPipeline:
    """Lightweight integration tests for the no-model retrieval pipeline."""

    def test_definition_chunks_survive_merge_and_generator_selection(self):
        """A simulated multi-part query keeps the relevant definition chunks."""
        from src.retrieval_selection import (
            merge_retrieval_runs,
            rerank_chunks_with_ids,
            rerank_with_query_overlap,
        )

        chunks = [
            (
                "A primary key is a candidate key chosen by the database "
                "designer to identify tuples in a relation."
            ),
            "A candidate key is a minimal superkey; no proper subset is also a superkey.",
            (
                "SQL includes a references privilege that permits users to "
                "declare foreign keys when creating relations."
            ),
            (
                "A foreign-key constraint states that values in one relation "
                "must appear as primary-key values in another relation. "
                "Attribute set A is called a foreign key from the referencing "
                "relation."
            ),
            "This chunk discusses storage pages and unrelated performance issues.",
        ]
        raw_runs = [
            ("Explain primary keys, candidate keys, and foreign keys.", [4, 2, 0, 1, 3]),
            ("What is a primary key?", [2, 0, 4]),
            ("What is a candidate key?", [4, 1, 0]),
            ("What is a foreign key?", [2, 3, 4]),
        ]

        retrieval_runs = []
        for question, candidates in raw_runs:
            ordered, scores = rerank_with_query_overlap(
                question,
                candidates,
                [1.0] * len(candidates),
                chunks,
            )
            retrieval_runs.append(
                {"question": question, "topk_idxs": ordered, "scores": scores}
            )

        selected, _ = merge_retrieval_runs(retrieval_runs, chunks, limit=4)
        _, sent_chunks = rerank_chunks_with_ids(
            raw_runs[0][0],
            selected,
            chunks,
            mode="",
            top_n=3,
            retrieval_runs=retrieval_runs,
        )

        sent_ids = {chunk["idx"] for chunk in sent_chunks}
        assert {0, 1, 3}.issubset(sent_ids)
        assert 2 not in sent_ids
