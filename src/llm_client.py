from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dotenv import load_dotenv

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

BATCH_METADATA_DIR = "cache/batches"
MAX_BATCH_SIZE_BYTES_OPENAI = 190 * 1024 * 1024
MAX_BATCH_SIZE_BYTES_ANTHROPIC = 200 * 1024 * 1024  # 256MB limit, use 200MB for safety
MAX_BATCH_REQUESTS_ANTHROPIC = 100_000
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


class Api(str, Enum):
    openrouter = "openrouter"
    openai = "openai"
    anthropic = "anthropic"


_ANTHROPIC_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-7",)


def _anthropic_supports_temperature(model: str) -> bool:
    return not model.startswith(_ANTHROPIC_NO_TEMPERATURE_PREFIXES)


def cache_key(messages: list[dict], model: str) -> str:
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


def find_latest_metadata(task_name: str | None = None) -> str:
    meta_dir = Path(BATCH_METADATA_DIR)
    if not meta_dir.exists():
        raise FileNotFoundError(f"No batch metadata directory found at {BATCH_METADATA_DIR}")

    candidates = sorted(meta_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No metadata files found in {BATCH_METADATA_DIR}")

    if task_name:
        for c in candidates:
            if c.stem.startswith(task_name):
                return str(c)
        raise FileNotFoundError(
            f"No metadata file found for task_name={task_name} in {BATCH_METADATA_DIR}"
        )

    return str(candidates[0])


def _extract_system_and_user_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_messages.append(m)
    return system, user_messages


class LLMClient:
    def __init__(
        self,
        api: str | Api = "anthropic",
        model: str = "claude-opus-4-6",
        max_tokens: int = 2048,
        temperature: float = 0,
    ):
        self.api = Api(api)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.api == Api.anthropic:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        elif self.api == Api.openai:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.api == Api.openrouter:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise ValueError(f"Unsupported API: {self.api}")

        self._client = client
        return client

    async def call(
        self,
        messages: list[dict],
        cache: dict,
        semaphore: asyncio.Semaphore,
    ) -> str:
        key = cache_key(messages, self.model)
        if key in cache:
            return cache[key]

        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    if self.api == Api.anthropic:
                        result = await self._call_anthropic(messages)
                    else:
                        result = await self._call_openai(messages)
                cache[key] = result
                return result
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"\nRetry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable: all retries should either return or raise")

    async def _call_anthropic(self, messages: list[dict]) -> str:
        client = cast(AsyncAnthropic, self._get_client())
        system, user_messages = _extract_system_and_user_messages(messages)
        typed_messages = cast(list[MessageParam], user_messages)
        kwargs: dict = {
            "model": self.model,
            "messages": typed_messages,
            "max_tokens": self.max_tokens,
        }
        if _anthropic_supports_temperature(self.model):
            kwargs["temperature"] = self.temperature
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        text = getattr(response.content[0], "text", None)
        assert text is not None, f"Unexpected content block type: {type(response.content[0])}"
        return text.strip()

    async def _call_openai(self, messages: list[dict]) -> str:
        client = cast(AsyncOpenAI, self._get_client())
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        if self.api == Api.openai:
            response = await client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
        else:
            response = await client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return (response.choices[0].message.content or "").strip()

    def submit(
        self,
        requests: list[tuple[str, list[dict]]],
        cache: dict,
        task_name: str,
    ) -> str:
        uncached = [
            (custom_id, msgs) for custom_id, msgs in requests
            if cache_key(msgs, self.model) not in cache
        ]
        n_cached = len(requests) - len(uncached)
        print(f"Cached: {n_cached}, to submit: {len(uncached)}")

        if not uncached:
            print("All requests are cached. Nothing to submit.")
            return ""

        if self.api == Api.anthropic:
            batches = self._submit_anthropic(uncached)
        elif self.api == Api.openai:
            batches = self._submit_openai(uncached)
        else:
            raise ValueError(f"Batch mode not supported for {self.api}")

        id_to_cache_key = {
            custom_id: cache_key(msgs, self.model) for custom_id, msgs in uncached
        }

        metadata = {
            "api": self.api.value,
            "batches": batches,
            "model": self.model,
            "task_name": task_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "custom_id_to_cache_key": id_to_cache_key,
            "n_total": len(requests),
            "n_cached": n_cached,
            "n_submitted": len(uncached),
        }

        os.makedirs(BATCH_METADATA_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{task_name}_{timestamp}.json"
        meta_path = os.path.join(BATCH_METADATA_DIR, filename)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {meta_path}")
        return meta_path

    def _submit_anthropic(self, uncached: list[tuple[str, list[dict]]]) -> list[dict]:
        from anthropic import Anthropic

        client = Anthropic()

        chunks = []
        current_chunk = []
        current_size = 0

        for custom_id, messages in uncached:
            system, user_messages = _extract_system_and_user_messages(messages)
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": user_messages,
            }
            if _anthropic_supports_temperature(self.model):
                params["temperature"] = self.temperature
            if system:
                params["system"] = system
            request = {"custom_id": custom_id, "params": params}
            request_size = len(json.dumps(request).encode("utf-8"))

            if current_chunk and (
                current_size + request_size > MAX_BATCH_SIZE_BYTES_ANTHROPIC
                or len(current_chunk) >= MAX_BATCH_REQUESTS_ANTHROPIC
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(request)
            current_size += request_size

        if current_chunk:
            chunks.append(current_chunk)

        batches = []
        for i, chunk in enumerate(chunks):
            batch = client.messages.batches.create(requests=chunk)
            batches.append({"batch_id": batch.id})
            print(f"  Batch {i+1}/{len(chunks)}: {batch.id} ({len(chunk)} requests)")

        return batches

    def _submit_openai(self, uncached: list[tuple[str, list[dict]]]) -> list[dict]:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        batches = []

        current_lines = []
        current_size = 0
        chunks = []

        for custom_id, messages in uncached:
            line = json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                },
            })
            line_size = len(line.encode("utf-8")) + 1
            if current_lines and current_size + line_size > MAX_BATCH_SIZE_BYTES_OPENAI:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_size = 0
            current_lines.append(line)
            current_size += line_size

        if current_lines:
            chunks.append("\n".join(current_lines))

        for i, jsonl_content in enumerate(chunks):
            jsonl_bytes = jsonl_content.encode("utf-8")
            input_file = client.files.create(
                file=("batch_input.jsonl", jsonl_bytes),
                purpose="batch",
            )
            batch = client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batches.append({"batch_id": batch.id, "input_file_id": input_file.id})
            size_mb = len(jsonl_bytes) / (1024 * 1024)
            n_lines = jsonl_content.count("\n") + 1
            print(f"  Batch {i+1}/{len(chunks)}: {batch.id} ({n_lines} requests, {size_mb:.0f}MB)")

        return batches

    @staticmethod
    def check(metadata_path: str | None = None, task_name: str | None = None) -> str:
        if metadata_path is None:
            metadata_path = find_latest_metadata(task_name)
        with open(metadata_path) as f:
            metadata = json.load(f)

        api = Api(metadata["api"])
        batches_info = metadata["batches"]

        print(f"Task: {metadata.get('task_name', 'unknown')}")
        print(f"Submitted: {metadata['n_submitted']} / Total: {metadata['n_total']}")
        print(f"Batches: {len(batches_info)}")

        if api == Api.anthropic:
            statuses = _check_anthropic(batches_info)
        elif api == Api.openai:
            statuses = _check_openai(batches_info)
        else:
            raise ValueError(f"Batch mode not supported for {api}")

        print(f"Metadata: {metadata_path}")

        if all(s in ("completed", "ended") for s in statuses):
            return "completed"
        if any(s in ("failed", "expired", "cancelled", "canceling") for s in statuses):
            return "failed"
        return "in_progress"

    @staticmethod
    def collect(metadata_path: str | None = None, task_name: str | None = None, cache: dict | None = None, cache_path: str | None = None) -> int:
        if metadata_path is None:
            metadata_path = find_latest_metadata(task_name)
        with open(metadata_path) as f:
            metadata = json.load(f)

        api = Api(metadata["api"])
        batches_info = metadata["batches"]
        id_to_cache_key = metadata["custom_id_to_cache_key"]

        if cache is None:
            if cache_path is None:
                raise ValueError("Either cache or cache_path must be provided")
            cache = load_cache(cache_path)

        if api == Api.anthropic:
            n_collected, n_failed = _collect_anthropic(batches_info, id_to_cache_key, cache)
        elif api == Api.openai:
            n_collected, n_failed = _collect_openai(batches_info, id_to_cache_key, cache)
        else:
            raise ValueError(f"Batch mode not supported for {api}")

        print(f"Collected: {n_collected}, Failed: {n_failed}")

        if cache_path:
            save_cache(cache, cache_path)
            print(f"Cache updated: {cache_path}")

        return n_collected


def _check_anthropic(batches_info: list[dict]) -> list[str]:
    from anthropic import Anthropic

    client = Anthropic()
    statuses = []
    for i, bi in enumerate(batches_info):
        batch = client.messages.batches.retrieve(bi["batch_id"])
        status = batch.processing_status
        counts = batch.request_counts
        if status == "ended" and counts.succeeded == 0 and counts.errored > 0:
            status = "failed"
        statuses.append(status)
        prefix = f"  [{i+1}/{len(batches_info)}] {batch.id}"
        print(f"{prefix}: {status}")
        print(f"    Processing: {counts.processing}, Succeeded: {counts.succeeded}, "
              f"Errored: {counts.errored}, Canceled: {counts.canceled}, Expired: {counts.expired}")
    return statuses


def _check_openai(batches_info: list[dict]) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    statuses = []
    for i, bi in enumerate(batches_info):
        batch = client.batches.retrieve(bi["batch_id"])
        statuses.append(batch.status)
        prefix = f"  [{i+1}/{len(batches_info)}] {batch.id}"
        print(f"{prefix}: {batch.status}")
        if batch.request_counts:
            rc = batch.request_counts
            print(f"    Completed: {rc.completed}, Failed: {rc.failed}, Total: {rc.total}")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"    [{err.code}] {err.message} (line={err.line}, param={err.param})")
    return statuses


def _collect_anthropic(
    batches_info: list[dict],
    id_to_cache_key: dict,
    cache: dict,
) -> tuple[int, int]:
    from anthropic import Anthropic

    client = Anthropic()
    n_collected = 0
    n_failed = 0

    error_samples: list[str] = []
    for bi in batches_info:
        batch = client.messages.batches.retrieve(bi["batch_id"])
        if batch.processing_status not in ("ended",):
            print(f"Batch {batch.id} status is '{batch.processing_status}', not ready for collection.")
            continue

        for result in client.messages.batches.results(bi["batch_id"]):
            custom_id = result.custom_id
            ck = id_to_cache_key.get(custom_id)
            if ck is None:
                continue
            if result.result.type == "succeeded":
                content = result.result.message.content
                text = getattr(content[0], "text", None) if content else None
                if text is not None:
                    cache[ck] = text.strip()
                    n_collected += 1
                else:
                    n_failed += 1
            else:
                n_failed += 1
                if len(error_samples) < 3:
                    detail = getattr(result.result, "error", result.result)
                    error_samples.append(f"{custom_id}: {detail}")

    if error_samples:
        print(f"First {len(error_samples)} errored requests:")
        for s in error_samples:
            print(f"  {s}")

    return n_collected, n_failed


def _collect_openai(
    batches_info: list[dict],
    id_to_cache_key: dict,
    cache: dict,
) -> tuple[int, int]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    n_collected = 0
    n_failed = 0

    for bi in batches_info:
        batch = client.batches.retrieve(bi["batch_id"])
        if batch.status not in ("completed", "failed", "expired"):
            print(f"Batch {batch.id} status is '{batch.status}', not ready for collection.")
            continue

        if batch.output_file_id:
            content = client.files.content(batch.output_file_id)
            for line in content.text.strip().split("\n"):
                if not line:
                    continue
                result_obj = json.loads(line)
                custom_id = result_obj["custom_id"]
                ck = id_to_cache_key.get(custom_id)
                if ck is None:
                    continue
                response_body = result_obj.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    text = (choices[0].get("message", {}).get("content") or "").strip()
                    cache[ck] = text
                    n_collected += 1
                else:
                    error = result_obj.get("error") or result_obj.get("response", {}).get("error")
                    print(f"No choices for {custom_id}: {error}")
                    n_failed += 1

        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            for line in error_content.text.strip().split("\n"):
                if not line:
                    continue
                error_result = json.loads(line)
                custom_id = error_result.get("custom_id", "unknown")
                error = error_result.get("error", {})
                print(f"Error for {custom_id}: {error}")
                n_failed += 1

    return n_collected, n_failed
