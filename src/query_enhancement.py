"""
Query enhancement techniques for improved retrieval (use only one):
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer for better retrieval
- Query Enrichment: LLM-based query expansion
"""
import textwrap
import re
from typing import Optional
from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning


_LEADING_SUBQUERY_JUNK = "\ufeff\u200b\u00bf\u00a1\"'`[](){}.,;:!?-_* "
_GENERATED_QUERY_LABEL_RE = re.compile(
    r"\b(?:output|answer|question|standalone question|rewritten query|rewritten question)\s*:\s*",
    re.IGNORECASE,
)


def clean_generated_query(text: str, fallback: Optional[str] = None) -> str:
    """Clean labels and accidental prompt echoes from an LLM-generated query."""
    cleaned = " ".join(str(text).split()).strip()
    cleaned = cleaned.replace("\u00c2\u00bf", "").strip()

    label_matches = list(_GENERATED_QUERY_LABEL_RE.finditer(cleaned))
    if label_matches:
        candidate = cleaned[label_matches[-1].end():].strip()
        if candidate:
            cleaned = candidate

    while True:
        stripped = _GENERATED_QUERY_LABEL_RE.sub("", cleaned, count=1).strip()
        if stripped == cleaned:
            break
        cleaned = stripped

    cleaned = cleaned.lstrip(_LEADING_SUBQUERY_JUNK).strip()
    if not cleaned and fallback is not None:
        return fallback

    if fallback is not None and len(cleaned) > max(len(fallback) * 2, len(fallback) + 80):
        return fallback

    return cleaned


def _clean_decomposed_question(text: str) -> str:
    """Normalize one LLM-generated sub-question."""
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.replace("\u00c2\u00bf", "").strip()
    cleaned = cleaned.lstrip(_LEADING_SUBQUERY_JUNK)
    cleaned = re.sub(r"^\d+\s*[\).\:-]\s*", "", cleaned).strip()

    for prefix in ("output:", "output"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" :")

    cleaned = cleaned.lstrip(_LEADING_SUBQUERY_JUNK)
    return cleaned


def generate_hypothetical_document(
    query: str,
    model_path: str,
    max_tokens: int = 100,
    **llm_kwargs
) -> str:
    """
    HyDE: Generate a hypothetical answer to improve retrieval quality.
    Concept: Hypothetical answers are semantically closer to actual documents than queries.
    Ref: https://arxiv.org/abs/2212.10496
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a database systems expert. Generate a concise, technical answer using precise database terminology.
        Write in the formal academic style of Database System Concepts (Silberschatz, Korth, Sudarshan).
        Use specific terms for: relational model concepts (relations, tuples, attributes, keys, schemas), 
        SQL and query languages, transactions (ACID properties, concurrency control, recovery), 
        storage structures (indexes, B+ trees), normalization (functional dependencies, normal forms), 
        and database design (E-R model, decomposition).
        Focus on definitions, mechanisms, and technical accuracy rather than examples.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Generate a precise and a concise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    hypothetical = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        **llm_kwargs
    )
    
    return hypothetical.strip()

def correct_query_grammar(
    query: str,
    model_path: str,
    **llm_kwargs
) -> str:
    """
    Corrects spelling and grammatical errors in the query to improve keyword matching.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that corrects search queries.
        Your task is to correct any spelling or grammatical errors in the user's query.
        Do not answer the question. Output ONLY the corrected query.
        <|im_end|>
        <|im_start|>user
        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    corrected_query = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=len(query.split()) * 2,
        temperature=0,
        **llm_kwargs
    )

    # If model returns empty or hallucinated long text, return original
    cleaned = corrected_query["choices"][0]["text"].strip()
    if not cleaned or len(cleaned) > len(query) * 2:
        return query

    return cleaned

def expand_query_with_keywords(
    query: str,
    model_path: str,
    max_tokens: int = 64,
    **llm_kwargs
) -> str:
    """
    Query Expansion: Generates related keywords and synonyms.
    This helps retrieval when the user uses different vocabulary than the documents.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search optimization expert.
        Generate 3 alternative versions of the user's query using synonyms and related technical terms.
        Output the alternative queries separated by newlines. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    expansion = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.5,
        **llm_kwargs
    )

    # Combine original query with expansion
    query_lines = [query]
    query_lines.extend([line.strip() for line in expansion["choices"][0]["text"].split('\n') if line.strip()])

    # Remove numbering if present
    query_lines = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in query_lines]

    return query_lines


def decompose_complex_query(
    query: str,
    model_path: str,
    max_sub_questions: int = 4,
    **llm_kwargs
) -> list[str]:
    """Split a multi-part question into smaller questions."""
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        Rewrite one database question into smaller standalone questions.

        Rules:
        - Return only the rewritten questions.
        - Put one question on each line.
        - Every line must end with ?.
        - Do not write labels, numbering, bullets, explanations, or quotes.
        - Do not write words like "Output" or "Answer".
        - Keep important database terms exactly as written.
        - Preserve necessary topic qualifiers in every line, such as "under snapshot isolation", "in SQL", or "for B+ trees".
        - Do not make vague subquestions that only make sense if another generated line is read first.
        - Cover the main parts of the original question.
        - Write at most {max_sub_questions} lines.
        - If the question is already simple, return it unchanged.
        <|im_end|>
        <|im_start|>user
        Complex Question: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=128,
        temperature=0.0,
        **llm_kwargs
    )

    # Split on ? because the local model sometimes puts multiple questions on one line.
    raw_text = output["choices"][0]["text"].replace("\r", " ").replace("\n", " ").strip()
    parts = raw_text.split("?")

    sub_questions = []
    for part in parts:
        cleaned = _clean_decomposed_question(part)
        if not cleaned:
            continue
        sub_questions.append(f"{cleaned}?")
        if len(sub_questions) >= max_sub_questions:
            break

    return sub_questions

def contextualize_query(
    query: str,
    history: list[dict],
    model_path: str,
    max_tokens: int = 128,
    **llm_kwargs
) -> str:
    """
    Rewrites a query to be standalone based on chat history.
    """
    if not history:
        return query

    # Format history into a compact string
    # We expect history to be list of dicts: [{"role": "user", "content": "..."}, ...]
    conversation_text = ""
    for turn in history[-4:]: # Only look at last 2 turns
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        conversation_text += f"{role}: {content}\n"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant. Your task is to rewrite the user's "Follow Up Input" to be a standalone question by replacing pronouns (it, they, this, that) with specific nouns from the "Chat History".
        
        Examples:
        History: 
        User: What is BCNF?
        Assistant: It is a normal form used in database normalization.
        Input: Why is it useful?
        Output: Why is BCNF useful?
        
        History:
        User: Explain the ACID properties.
        Assistant: ACID stands for Atomicity, Consistency, Isolation, Durability.
        Input: Give me an example of the first one.
        Output: Give me an example of Atomicity.

        History:
        User: Who created Python?
        Assistant: Guido van Rossum.
        Input: what is sql?
        Output: what is sql?
        <|im_end|>
        <|im_start|>user
        Chat History:
        {conversation_text}
        
        Follow Up Input: {query}
        
        Output:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs
    )

    rewritten = clean_generated_query(output["choices"][0]["text"], fallback=query)

    # If model hallucinates or errors, fall back to original query
    if not rewritten or len(rewritten) > max(len(query) * 2, len(query) + 80):
        return query

    return rewritten
