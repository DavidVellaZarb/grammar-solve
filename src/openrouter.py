import asyncio
import hashlib
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def cache_key(messages: list[ChatCompletionMessageParam], model: str) -> str:
    key_data = {"messages": messages, "model": model}
    serialized = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    cache: dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 1024,
) -> str:
    key = cache_key(messages, model)
    if key in cache:
        return cache[key]

    async with semaphore:
        token_param = (
            {"max_completion_tokens": max_tokens}
            if "openai.com" in str(client.base_url)
            else {"max_tokens": max_tokens}
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            **token_param,
        )
    result = (response.choices[0].message.content or "").strip()
    cache[key] = result
    return result
