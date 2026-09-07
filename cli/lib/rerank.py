import json
from time import sleep
from typing import Literal, NotRequired

from .search_utils import SearchResult
from .llm_client import client, model, provider
from sentence_transformers import CrossEncoder


class RerankedSearchResult(SearchResult, total=False):
    individual_score: NotRequired[float]
    batch_rank: NotRequired[int]
    crossencoder_score: NotRequired[float]

def llm_rerank_individual(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    scored_docs: list[RerankedSearchResult] = []

    for index, doc in enumerate(documents):
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Identify every important constraint in the query before scoring.
        Score 9-10 only when the movie directly satisfies nearly all constraints.
        Penalize movies that match only incidental words or a single constraint.

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict movie search relevance evaluator.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                score_text = (response.choices[0].message.content or "").strip()
                score = float(score_text.removeprefix("assistant").strip())
                if not 0 <= score <= 10:
                    raise ValueError("Re-rank score must be between 0 and 10")
                break
            except Exception:
                sleep(3 if provider == "openrouter" else 1)

        scored_docs.append({**doc, "individual_score": score})
        if provider == "openrouter" and index < len(documents) - 1:
            sleep(3)

    scored_docs.sort(
        key=lambda result: (result["individual_score"], result["score"]),
        reverse=True,
    )
    return scored_docs[:limit]


def llm_rerank_batch(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    # Full plots can exceed tens of thousands of characters; a synopsis is
    # enough for ranking and keeps local-model batch latency manageable.
    doc_list_str = "\n\n".join(
        f"ID: {doc['id']}\nTitle: {doc['title']}\nDescription: {doc['document'][:1000]}"
        for doc in documents
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    candidate_ids = [doc["id"] for doc in documents]
    expected_ids = set(candidate_ids)
    request_options = {"extra_body": {"think": False}} if provider == "ollama" else {}

    last_error: Exception | None = None
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict movie search relevance ranker.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                **request_options,
            )
            response_text = (response.choices[0].message.content or "").strip()
            json_text = response_text.removeprefix("assistant").strip()
            array_start = json_text.find("[")
            array_end = json_text.rfind("]")
            if array_start == -1 or array_end < array_start:
                raise ValueError(
                    "Batch ranking response did not contain a JSON array: "
                    f"{response_text[:200]!r}"
                )
            parsed_ranking = json.loads(json_text[array_start : array_end + 1])
            ranked_ids = (
                parsed_ranking.get("ranked_ids")
                if isinstance(parsed_ranking, dict)
                else parsed_ranking
            )
            if (
                not isinstance(ranked_ids, list)
                or any(type(doc_id) is not int for doc_id in ranked_ids)
            ):
                raise ValueError("Batch ranking must be a JSON array of integer movie IDs")

            seen_ids: set[int] = set()
            valid_ranked_ids: list[int] = []
            for doc_id in ranked_ids:
                if doc_id in expected_ids and doc_id not in seen_ids:
                    valid_ranked_ids.append(doc_id)
                    seen_ids.add(doc_id)
            if not valid_ranked_ids:
                raise ValueError("Batch ranking did not contain any candidate movie IDs")
            valid_ranked_ids.extend(
                doc_id for doc_id in candidate_ids if doc_id not in seen_ids
            )
            ranked_ids = valid_ranked_ids
            break
        except Exception as error:
            last_error = error
            sleep(3 if provider == "openrouter" else 1)
    else:
        raise RuntimeError(
            f"Batch reranking failed after 5 attempts using {provider}/{model}: "
            f"{last_error}"
        ) from last_error

    rank_by_id = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, 1)}
    ranked_docs: list[RerankedSearchResult] = [
        {**doc, "batch_rank": rank_by_id[doc["id"]]} for doc in documents
    ]
    ranked_docs.sort(key=lambda result: result["batch_rank"])
    return ranked_docs[:limit]

def llm_rerank_cross_encoder(query, documents, limit):
    ranked_docs: list[RerankedSearchResult] = []
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        ranked_docs.append({**doc, "crossencoder_score": score})

    ranked_docs.sort(
        key=lambda result: (result["crossencoder_score"], result["score"]),
        reverse=True,
    )
    return ranked_docs[:limit]



def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch"] = "individual",
    limit: int = 5,
) -> list[SearchResult] | list[RerankedSearchResult]:
    match method:
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "batch":
            return llm_rerank_batch(query, documents, limit)
        case "cross_encoder":
            return llm_rerank_cross_encoder(query, documents, limit)
        case _:
            raise ValueError(f"Invalid reranking method: {method}")
    
