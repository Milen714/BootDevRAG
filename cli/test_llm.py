import os
from dotenv import load_dotenv
from openai import OpenAI


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