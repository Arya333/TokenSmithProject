# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from src.cache import get_cache
from src.config import RAGConfig
from src.generator import answer, dedupe_generated_text, double_answer
from src.index_builder import build_index
from src.index_updater import add_to_index
from src.instrumentation.logging import get_logger
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import (
    contextualize_query,
    decompose_complex_query,
    generate_hypothetical_document,
)
from src.ranking.ranker import EnsembleRanker
from src.ranking.reranker import rerank
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    filter_retrieved_chunks,
    get_page_numbers,
    load_artifacts,
)

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat", "add-chapters"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="use a partial index stored in 'index/partial_sections' instead of 'index/sections'",
    )
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument("--system_prompt_mode", choices=["baseline", "tutor", "concise", "detailed"], default="baseline")

    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")
    indexing_group.add_argument(
        "--chapters",
        nargs="+",
        type=int,
        help="a list of chapter numbers to index (e.g., --chapters 3 4 5)",
    )
    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers",
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory(partial=args.partial)

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        embedding_model_context_window=cfg.embedding_model_context_window,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
        chapters_to_index=args.chapters,
    )


def run_add_chapters_mode(args: argparse.Namespace, cfg: RAGConfig):
    """Handles the logic for adding chapters to an existing index."""
    if not args.chapters:
        print("Please provide a list of chapters to add using the --chapters argument.")
        return

    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory(partial=True)

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    add_to_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        embedding_model_context_window=cfg.embedding_model_context_window,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        chapters_to_add=args.chapters,
        use_headings=args.embed_with_headings,
    )
    print("Successfully added chapters to the index.")


def use_indexed_chunks(question: str, chunks: list, cfg: RAGConfig, args: argparse.Namespace) -> tuple[list, list[int]]:
    try:
        artifacts_dir = cfg.get_artifacts_directory(partial=getattr(args, "partial", False))
        map_path = cfg.get_page_to_chunk_map_path(artifacts_dir, args.index_prefix)
        with open(map_path, "r") as f:
            page_to_chunk_map = json.load(f)
        with open("data/extracted_index.json", "r") as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return [], []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    return [chunks[cid] for cid in chunk_ids], list(chunk_ids)


def build_retrieval_queries(question: str, cfg: RAGConfig) -> list[str]:
    """Build the list of queries we will retrieve on."""
    queries = [question]

    if not cfg.use_query_decomposition:
        return queries

    try:
        sub_questions = decompose_complex_query(
            question,
            cfg.gen_model,
            max_sub_questions=cfg.max_sub_questions,
        )
    except Exception as e:
        print(f"Warning: Failed to decompose query: {e}. Using original query.")
        return queries

    seen = {question.strip().lower()}
    for sub_question in sub_questions:
        cleaned = sub_question.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        queries.append(cleaned)
        seen.add(lowered)
        if len(queries) >= cfg.max_sub_questions + 1:
            break

    return queries


def build_sub_query_ranker(cfg: RAGConfig, retrievers: list, default_ranker: Any) -> Any:
    """Use a small FAISS + BM25 mix for sub-questions if BM25 is off."""
    if cfg.ranker_weights.get("bm25", 0) > 0:
        return default_ranker

    retriever_names = {retriever.name for retriever in retrievers}
    if "faiss" not in retriever_names or "bm25" not in retriever_names:
        return default_ranker

    return EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights={"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
        rrf_k=int(cfg.rrf_k),
    )


def retrieve_chunks_for_query(
    query: str,
    cfg: RAGConfig,
    retrievers: list,
    ranker: Any,
    chunks: list,
) -> tuple[list[int], list[float], Dict[str, Dict[int, float]], str]:
    """Run retrieval once for one query."""
    retrieval_query = query
    if cfg.use_hyde:
        retrieval_query = generate_hypothetical_document(
            query,
            cfg.gen_model,
            max_tokens=cfg.hyde_max_tokens,
        )

    pool_n = max(cfg.num_candidates, cfg.top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)

    ordered, scores = ranker.rank(raw_scores=raw_scores)
    topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
    return topk_idxs, scores[: len(topk_idxs)], raw_scores, retrieval_query


def merge_chunk_lists(chunk_lists: list[list[int]], limit: int) -> list[int]:
    """Round-robin merge so one query does not fill every slot."""
    merged = []
    seen = set()
    positions = [0] * len(chunk_lists)

    while len(merged) < limit:
        added_any = False
        for list_idx, chunk_list in enumerate(chunk_lists):
            while positions[list_idx] < len(chunk_list):
                chunk_idx = chunk_list[positions[list_idx]]
                positions[list_idx] += 1
                if chunk_idx in seen:
                    continue
                merged.append(chunk_idx)
                seen.add(chunk_idx)
                added_any = True
                break

            if len(merged) >= limit:
                break

        if not added_any:
            break

    return merged


def merge_raw_scores(retrieval_runs: list[dict]) -> Dict[str, Dict[int, float]]:
    """Keep the strongest observed score per retriever/chunk across subquery runs."""
    merged: Dict[str, Dict[int, float]] = {}
    for run in retrieval_runs:
        for retriever_name, score_map in run["raw_scores"].items():
            merged_scores = merged.setdefault(retriever_name, {})
            for idx, score in score_map.items():
                if idx not in merged_scores or score > merged_scores[idx]:
                    merged_scores[idx] = score
    return merged


def build_chunks_info(topk_idxs: list[int], chunks: list, raw_scores: Dict[str, Dict[int, float]]) -> list[Dict[str, Any]]:
    faiss_scores = raw_scores.get("faiss", {})
    bm25_scores = raw_scores.get("bm25", {})
    index_scores = raw_scores.get("index_keywords", {})

    faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)
    bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
    index_ranked = sorted(index_scores.keys(), key=lambda i: index_scores[i], reverse=True)

    faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
    bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
    index_ranks = {idx: rank + 1 for rank, idx in enumerate(index_ranked)}

    chunks_info = []
    for rank, idx in enumerate(topk_idxs, 1):
        chunks_info.append(
            {
                "rank": rank,
                "chunk_id": idx,
                "content": chunks[idx],
                "faiss_score": faiss_scores.get(idx, 0),
                "faiss_rank": faiss_ranks.get(idx, 0),
                "bm25_score": bm25_scores.get(idx, 0),
                "bm25_rank": bm25_ranks.get(idx, 0),
                "index_score": index_scores.get(idx, 0),
                "index_rank": index_ranks.get(idx, 0),
            }
        )
    return chunks_info


def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """Run a single query through the pipeline."""
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]

    ranked_chunks: List[str] = []
    retrieved_chunks: List[str] = []
    retrieved_sources: List[str] = []
    topk_idxs: List[int] = []
    log_scores: List[float] = []
    log_info = dict(additional_log_info or {})

    chunks_info = None
    hyde_query = None
    raw_scores: Dict[str, Dict[int, float]] = {}

    cache = get_cache(cfg)
    normalized_question = cache.normalize_question(question)
    config_cache_key = cache.make_config_key(cfg, args, golden_chunks)
    question_embedding = cache.compute_embedding(normalized_question, retrievers, cfg.embed_model)
    semantic_hit = cache.lookup(config_cache_key, question_embedding, normalized_question)

    if semantic_hit is not None:
        ans = semantic_hit.get("answer", "")
        if is_test_mode:
            return ans, semantic_hit.get("chunks_info"), semantic_hit.get("hyde_query")
        console.print("Using cached answer")
        render_final_answer(console, ans)
        return ans

    if golden_chunks and cfg.use_golden_chunks:
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks, cfg, args)
        retrieved_chunks = ranked_chunks[:]
        retrieved_sources = [sources[i] for i in topk_idxs if 0 <= i < len(sources)]
        log_scores = [0.0] * len(topk_idxs)
    else:
        retrieval_queries = build_retrieval_queries(question, cfg)
        retrieval_runs = []
        sub_query_ranker = build_sub_query_ranker(cfg, retrievers, ranker)

        for query_idx, retrieval_question in enumerate(retrieval_queries):
            current_ranker = ranker if query_idx == 0 else sub_query_ranker
            query_topk, query_scores, query_raw_scores, actual_retrieval_query = retrieve_chunks_for_query(
                retrieval_question,
                cfg,
                retrievers,
                current_ranker,
                chunks,
            )
            retrieval_runs.append(
                {
                    "question": retrieval_question,
                    "retrieval_query": actual_retrieval_query,
                    "topk_idxs": query_topk,
                    "scores": query_scores,
                    "raw_scores": query_raw_scores,
                }
            )

        if len(retrieval_runs) == 1:
            run = retrieval_runs[0]
            topk_idxs = run["topk_idxs"]
            log_scores = run["scores"]
            raw_scores = run["raw_scores"]
            hyde_query = run["retrieval_query"] if cfg.use_hyde else None
        else:
            topk_idxs = merge_chunk_lists([run["topk_idxs"] for run in retrieval_runs], cfg.top_k)
            raw_scores = merge_raw_scores(retrieval_runs)

            best_score_by_chunk = {}
            for run in retrieval_runs:
                for idx, score in zip(run["topk_idxs"], run["scores"]):
                    if idx not in best_score_by_chunk or score > best_score_by_chunk[idx]:
                        best_score_by_chunk[idx] = score

            log_scores = [best_score_by_chunk.get(idx, 0.0) for idx in topk_idxs]
            log_info["used_query_decomposition"] = True
            log_info["target_sub_question_count"] = len(retrieval_queries) - 1
            log_info["retrieval_queries"] = retrieval_queries
            log_info["sub_query_ranker_weights"] = getattr(sub_query_ranker, "weights", {})
            log_info["sub_query_results"] = [
                {
                    "question": run["question"],
                    "top_idxs": run["topk_idxs"],
                }
                for run in retrieval_runs
            ]

        retrieved_chunks = [chunks[i] for i in topk_idxs]
        retrieved_sources = [sources[i] for i in topk_idxs]
        ranked_chunks = retrieved_chunks[:]

        if is_test_mode:
            chunks_info = build_chunks_info(topk_idxs, chunks, raw_scores)

        ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)

    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    use_double = getattr(args, "double_prompt", False) or cfg.use_double_prompt

    if use_double:
        stream_iter = double_answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        ans = dedupe_generated_text("".join(stream_iter))
    else:
        ans = render_streaming_ans(console, stream_iter)

        meta = artifacts.get("meta", [])
        page_nums = get_page_numbers(topk_idxs, meta)
        logger.save_chat_log(
            query=question,
            config_state=cfg.get_config_state(),
            ordered_scores=log_scores,
            chat_request_params={
                "system_prompt": system_prompt,
                "max_tokens": cfg.max_gen_tokens,
            },
            top_idxs=topk_idxs,
            chunks=retrieved_chunks,
            sources=retrieved_sources,
            page_map=page_nums,
            full_response=ans,
            top_k=len(topk_idxs),
            additional_log_info=log_info,
        )

    cache_payload = {
        "answer": ans,
        "chunks_info": chunks_info,
        "hyde_query": hyde_query,
        "chunk_indices": topk_idxs,
    }
    if question_embedding is None:
        question_embedding = cache.compute_embedding(normalized_question, retrievers, cfg.embed_model)
    cache.store(config_cache_key, normalized_question, question_embedding, cache_payload)

    if is_test_mode:
        return ans, chunks_info, hyde_query

    return ans


def render_streaming_ans(console, stream_iter):
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
    return ans


def render_final_answer(console, ans):
    if not console:
        raise ValueError("Console must be non null for rendering.")
    console.print(
        "\n[bold cyan]==================== START OF ANSWER ===================[/bold cyan]\n"
    )
    console.print(Markdown(ans))
    console.print(
        "\n[bold cyan]===================== END OF ANSWER ====================[/bold cyan]\n"
    )


def get_keywords(question: str) -> list:
    """Simple keyword extraction from the question."""
    stopwords = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "for",
        "a",
        "an",
        "and",
        "or",
        "in",
        "to",
        "of",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "are",
        "was",
        "what",
    }
    words = question.lower().split()
    return [word.strip(".,!?()[]") for word in words if word not in stopwords]


def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    logger = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory(partial=args.partial)
        cfg.page_to_chunk_map_path = cfg.get_page_to_chunk_map_path(artifacts_dir, args.index_prefix)
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))

        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k),
        )
        print("Loaded retrievers and initialized ranker.")
        artifacts = {"chunks": chunks, "sources": sources, "retrievers": retrievers, "ranker": ranker, "meta": meta}
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history = []
    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        print("CHAT HISTORY:", chat_history)
        try:
            additional_log_info = {}
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                    print(f"Contextualized Query: {effective_q}")
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q

            ans = get_answer(
                effective_q,
                cfg,
                args,
                logger,
                console,
                artifacts=artifacts,
                additional_log_info=additional_log_info,
            )

            try:
                user_turn = {"role": "user", "content": q}
                assistant_turn = {"role": "assistant", "content": ans}
                chat_history += [user_turn, assistant_turn]
            except Exception as e:
                print(f"Warning: Failed to update chat history: {e}")

            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2 :]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
            break


def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)
    elif args.mode == "add-chapters":
        run_add_chapters_mode(args, cfg)


if __name__ == "__main__":
    main()
