import json
import os
import time
from pydoc import doc
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal, NotRequired
from lib.search_utils import SearchResult

class RerankedSearchResult(SearchResult, total=False):
    individual_score: NotRequired[int]

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# messages = [
#     {
#         "role": "user",
#         "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
#     }
# ]

# response = client.chat.completions.create(
#     model="openrouter/free",
#     messages=messages,
# )

# print(response.choices[0].message.content)
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Response tokens: {response.usage.completion_tokens}")

def spell_correction(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Fix any spelling errors in the user-provided movie search query below.
                        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                        Preserve punctuation and capitalization unless a change is required for a typo fix.
                        If there are no spelling errors, or if you're unsure, output the original query unchanged.
                        Output only the final query text, nothing else.
                        User query: "{query}"
                        """,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    corrected_query = response.choices[0].message.content.strip()
    return corrected_query

def rewrite_query(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

def expand_query(query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            User query: "{query}"
            """,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    expanded_query = response.choices[0].message.content.strip()
    return expanded_query

def rerank_individual(doc: dict, query: str) -> float:
    messages = [
        {
            "role": "user",
            "content": f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:""",
        }
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )
            result = response.choices[0].message.content.strip()
            score = float(result)
            if not 0 <= score <= 10:
                raise ValueError("Re-rank score must be between 0 and 10")
            return score
        except Exception:
            time.sleep(3)

def batch_rerank(doc_list_str, query: str) -> list[float]:
    messages = [
        {
            "role": "user",
            "content": f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return the movie IDs in order of relevance, best match first.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:""",
        }
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )
            results = json.loads(response.choices[0].message.content.strip())
            ranked_ids = [int(doc_id) for doc_id in results]
            return ranked_ids
        except Exception:
            time.sleep(3)

def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"] = "batch",
    limit: int = 5,
) -> list[SearchResult] | list[RerankedSearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    else:
        return documents[:limit]
