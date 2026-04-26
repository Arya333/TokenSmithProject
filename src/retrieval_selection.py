"""Generic helpers for choosing retrieval and generator context chunks."""

import re
from typing import Dict, Optional


STOPWORDS = {
    "about", "also", "and", "are", "assuming", "can", "does", "for", "from",
    "how", "into", "is", "it", "its", "mean", "means", "of", "or", "still",
    "that", "the", "their", "them", "they", "this", "to", "under", "what",
    "when", "where", "which", "why", "with",
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def variants(term: str) -> set[str]:
    choices = {term}
    if term.endswith("ies") and len(term) > 4:
        choices.add(f"{term[:-3]}y")
    elif term.endswith("s") and len(term) > 3:
        choices.add(term[:-1])
    elif len(term) >= 3:
        choices.add(f"{term}s")
    return choices


def content_terms(text: str) -> list[str]:
    terms = []
    seen = set()
    for token in normalize(text).split():
        if len(token) < 3 or token in STOPWORDS:
            continue
        for term in variants(token):
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms


def exact_count(text: str, phrase: str) -> int:
    return len(re.findall(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def phrase_variants(phrase: str) -> set[str]:
    tokens = normalize(phrase).split()
    if not tokens:
        return set()
    return {" ".join(tokens[:-1] + [last]) for last in variants(tokens[-1])}


def definition_target(query: str) -> str:
    normalized = normalize(query)
    patterns = (
        r"^what (?:is|are) (?:a |an |the )?(?P<target>.+)$",
        r"^define (?:a |an |the )?(?P<target>.+)$",
        r"^how (?:is|are) (?:a |an |the )?(?P<target>.+?) defined$",
        r"^(?:what does|what do) (?:a |an |the )?(?P<target>.+?) mean$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            return re.sub(r"\b(?:mean|means|defined)\b$", "", match.group("target")).strip()
    return ""


def is_definition_query(query: str) -> bool:
    return bool(definition_target(query))


def score_definition_cues(query: str, chunk: str) -> float:
    target = definition_target(query)
    if not target:
        return 0.0

    best = 0.0
    for sentence in re.split(r"(?<=[.!?])\s+", chunk.lower()):
        normalized = normalize(sentence)
        for phrase in phrase_variants(target):
            escaped = re.escape(phrase)
            target_defines = (
                rf"(?<![a-z0-9]){escaped}(?![a-z0-9])\s+"
                r"(?:is|are|means|mean|refers to|denotes|denote|"
                r"is called|are called|is defined as|are defined as)\b"
            )
            called_target = (
                r"\b(?:called|known as|termed|defined as)\s+"
                rf"(?:a|an|the)?\s*(?<![a-z0-9]){escaped}(?![a-z0-9])"
            )
            example_target = (
                r"\b(?:is|are)\s+"
                rf"(?:a|an|the)?\s*(?<![a-z0-9]){escaped}(?![a-z0-9])"
            )
            term_target = rf"\bterm\s+(?<![a-z0-9]){escaped}(?![a-z0-9])\s+to\s+denote\b"

            if re.search(target_defines, normalized) or re.search(called_target, normalized) or re.search(term_target, normalized):
                best = max(best, 1.0)
            elif re.search(example_target, normalized):
                best = max(best, 0.5)
    return best


def overlap_score(query: str, chunk: str) -> float:
    terms = content_terms(query)
    if not terms:
        return 0.0

    normalized_chunk = normalize(chunk)
    matches = [exact_count(normalized_chunk, term) for term in terms]
    coverage = sum(1 for match in matches if match) / len(terms)
    frequency = sum(min(match, 5) for match in matches) / (len(terms) * 5)
    return (0.75 * coverage) + (0.25 * frequency)


def score_retrieval_candidate(query: str, rank: int, chunk: str) -> float:
    rank_score = 1.0 / (rank + 1)
    lexical = overlap_score(query, chunk)
    definition = score_definition_cues(query, chunk)
    if is_definition_query(query):
        return (0.20 * rank_score) + (0.45 * lexical) + (0.35 * definition)
    return (0.40 * rank_score) + (0.60 * lexical)


def rerank_with_query_overlap(
    query: str,
    ordered: list[int],
    scores: list[float],
    chunks: list[str],
) -> tuple[list[int], list[float]]:
    rescored = [
        (idx, score_retrieval_candidate(query, rank, chunks[idx]))
        for rank, idx in enumerate(ordered)
    ]
    rescored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in rescored], [score for _, score in rescored]


def section_key(chunk: str) -> str:
    return normalize(chunk.split(" Content:", 1)[0])[:220]


def diversity_adjusted_score(score: float, idx: int, selected: set[int], chunks: list[str]) -> float:
    if not selected:
        return score
    if any(section_key(chunks[idx]) == section_key(chunks[old_idx]) for old_idx in selected):
        return score * 0.9
    return score


def best_unselected_candidate(candidates: list[dict], selected: set[int], chunks: list[str]) -> Optional[dict]:
    available = [candidate for candidate in candidates if candidate["idx"] not in selected]
    if not available:
        return None
    return max(
        available,
        key=lambda candidate: diversity_adjusted_score(candidate["score"], candidate["idx"], selected, chunks),
    )


def run_candidates(run: dict, chunks: list[str]) -> list[dict]:
    candidates = []
    for rank, idx in enumerate(run["topk_idxs"]):
        score = run["scores"][rank] if rank < len(run["scores"]) else score_retrieval_candidate(run["question"], rank, chunks[idx])
        candidates.append({"idx": idx, "score": float(score), "query": run["question"]})
    return candidates


def merge_retrieval_runs(retrieval_runs: list[dict], chunks: list[str], limit: int) -> tuple[list[int], list[float]]:
    """Choose one strong chunk per subquery, then fill remaining slots globally."""
    selected: list[dict] = []
    selected_idxs: set[int] = set()
    per_run_candidates = [run_candidates(run, chunks) for run in retrieval_runs]

    for candidates in per_run_candidates[1:]:
        if len(selected) >= limit:
            break
        candidate = best_unselected_candidate(candidates, selected_idxs, chunks)
        if candidate:
            selected.append(candidate)
            selected_idxs.add(candidate["idx"])

    best_by_idx: dict[int, dict] = {}
    for candidates in per_run_candidates:
        for candidate in candidates:
            old = best_by_idx.get(candidate["idx"])
            if old is None or candidate["score"] > old["score"]:
                best_by_idx[candidate["idx"]] = candidate

    while len(selected) < limit:
        candidate = best_unselected_candidate(list(best_by_idx.values()), selected_idxs, chunks)
        if candidate is None:
            break
        selected.append(candidate)
        selected_idxs.add(candidate["idx"])

    return [item["idx"] for item in selected], [item["score"] for item in selected]


def run_score(run: dict, idx: int) -> float:
    for rank, run_idx in enumerate(run.get("topk_idxs", [])):
        if run_idx == idx and rank < len(run.get("scores", [])):
            return float(run["scores"][rank])
    return 0.0


def best_generator_coverage_item(
    run: dict,
    item_by_idx: Dict[int, dict],
    selected_idxs: set[int],
    pool_size: int,
    max_forced_rerank_rank: int,
) -> Optional[dict]:
    query = run.get("question", "")
    candidate_idxs = run.get("topk_idxs", [])[:pool_size]
    if not any(idx in item_by_idx and idx not in selected_idxs for idx in candidate_idxs):
        candidate_idxs = run.get("topk_idxs", [])

    def score(idx: int) -> tuple[float, float, int]:
        item = item_by_idx[idx]
        if is_definition_query(query):
            return (run_score(run, idx), 1.0 / item["rerank_rank"], -candidate_idxs.index(idx))
        return (1.0 / item["rerank_rank"], run_score(run, idx), -candidate_idxs.index(idx))

    available = [
        idx
        for idx in candidate_idxs
        if idx in item_by_idx
        and idx not in selected_idxs
        and item_by_idx[idx]["rerank_rank"] <= max_forced_rerank_rank
    ]
    if not available:
        return None
    idx = max(available, key=score)
    return {**item_by_idx[idx], "selection_reason": "subquery_coverage", "coverage_query": query}


def select_generator_chunks(
    ranked_items: list[dict],
    retrieval_runs: Optional[list[dict]],
    top_n: int,
) -> list[dict]:
    """Keep generator context diverse by covering subqueries before global fill."""
    if top_n <= 0:
        return []
    if not retrieval_runs or len(retrieval_runs) <= 1:
        return [{**item, "selection_reason": "global_rerank"} for item in ranked_items[:top_n]]

    selected: list[dict] = []
    selected_idxs: set[int] = set()
    item_by_idx = {item["idx"]: item for item in ranked_items}
    pool_size = max(top_n + 1, 6)
    max_forced_rerank_rank = max(top_n + 2, int(top_n * 1.5))

    for run in retrieval_runs[1:]:
        coverage_pool = set(run.get("topk_idxs", [])[:pool_size])
        already_covering = next((item for item in selected if item["idx"] in coverage_pool), None)
        if already_covering:
            already_covering.setdefault("covered_queries", []).append(run.get("question", ""))
            continue

        if len(selected) >= top_n:
            break
        item = best_generator_coverage_item(run, item_by_idx, selected_idxs, pool_size, max_forced_rerank_rank)
        if item:
            item["covered_queries"] = [run.get("question", "")]
            selected.append(item)
            selected_idxs.add(item["idx"])

    for item in ranked_items:
        if len(selected) >= top_n:
            break
        if item["idx"] not in selected_idxs:
            selected.append({**item, "selection_reason": "global_rerank"})
            selected_idxs.add(item["idx"])

    return selected


def rerank_chunks_with_ids(
    question: str,
    topk_idxs: list[int],
    chunks: list[str],
    mode: str,
    top_n: int,
    retrieval_runs: Optional[list[dict]] = None,
) -> tuple[list, list[dict]]:
    selected_chunks = [chunks[idx] for idx in topk_idxs]
    if mode == "cross_encoder":
        from src.ranking.reranker import rerank
        reranked = rerank(question, selected_chunks, mode=mode, top_n=len(selected_chunks))
    else:
        reranked = selected_chunks

    chunk_to_indices: Dict[str, list[int]] = {}
    for idx, chunk in zip(topk_idxs, selected_chunks):
        chunk_to_indices.setdefault(chunk, []).append(idx)

    ranked_items = []
    for rerank_rank, item in enumerate(reranked, 1):
        chunk_text = item[0] if isinstance(item, tuple) else item
        rerank_score = float(item[1]) if isinstance(item, tuple) else None
        idxs = chunk_to_indices.get(chunk_text, [])
        if not idxs:
            continue
        ranked_items.append(
            {
                "idx": idxs.pop(0),
                "chunk": chunk_text,
                "ranked_item": item,
                "rerank_score": rerank_score,
                "rerank_rank": rerank_rank,
            }
        )

    selected_items = select_generator_chunks(ranked_items, retrieval_runs, top_n)
    ranked_chunks = [item["ranked_item"] for item in selected_items]
    sent_chunks = []
    for rank, item in enumerate(selected_items, 1):
        sent_chunks.append(
            {
                "rank": rank,
                "idx": item["idx"],
                "chunk": item["chunk"],
                "rerank_score": item["rerank_score"],
                "rerank_rank": item["rerank_rank"],
                "selection_reason": item.get("selection_reason", "global_rerank"),
                "coverage_query": item.get("coverage_query"),
                "covered_queries": item.get("covered_queries", []),
            }
        )
    return ranked_chunks, sent_chunks
