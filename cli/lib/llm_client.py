import os
import socket
import struct

from dotenv import load_dotenv
from openai import OpenAI


def _wsl_gateway() -> str | None:
    if "WSL_INTEROP" not in os.environ:
        return None

    try:
        with open("/proc/net/route", encoding="ascii") as routes:
            next(routes)
            for route in routes:
                fields = route.split()
                if fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, IndexError, ValueError):
        return None

    return None


load_dotenv()

provider = os.getenv("LLM_PROVIDER", "ollama").lower()
model_override = os.getenv("LLM_MODEL")

if provider == "ollama":
    gateway = _wsl_gateway()
    default_base_url = (
        f"http://{gateway}:11434/v1"
        if gateway
        else "http://localhost:11434/v1"
    )
    base_url = os.getenv("OLLAMA_BASE_URL", default_base_url)
    if gateway:
        base_url = base_url.replace("localhost", gateway, 1).replace(
            "127.0.0.1", gateway, 1
        )
    model = model_override or os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    client = OpenAI(base_url=base_url, api_key="ollama")
elif provider == "openrouter":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required when "
            "LLM_PROVIDER=openrouter"
        )
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = model_override or os.getenv("OPENROUTER_MODEL", "openrouter/free")
    client = OpenAI(base_url=base_url, api_key=api_key)
else:
    raise RuntimeError("LLM_PROVIDER must be either 'ollama' or 'openrouter'")
