from typing import Literal

from .llm_client import client, model


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided document search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    corrected = (response.choices[0].message.content or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided policy-document query below to be more specific and searchable.

    Consider:
    - Preserve the user's policy, safety, regulation, or technology concepts
    - Prefer terminology likely to occur in research and policy documents
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    rewritten = (response.choices[0].message.content or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided policy-document query below with related terms.

    Add synonyms and related concepts that might appear in policy and research documents.
    Keep expansions relevant and focused.
    Do not repeat the original query.
    Output only the additional terms; they will be appended to the original query.

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    expanded_terms = (response.choices[0].message.content or "").strip().strip('"')
    if expanded_terms.casefold().startswith(query.casefold()):
        expanded_terms = expanded_terms[len(query):].lstrip(" \n:->\"'")
    return f"{query} {expanded_terms}".strip()


def enhance_query(
    query: str, method: Literal["spell", "rewrite", "expand"] | None = None
) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
