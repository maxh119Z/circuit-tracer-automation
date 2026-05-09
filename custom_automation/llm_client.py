"""
Provider-agnostic LLM client factory used by validate_neuronpedia_groups.py.

LLM_PROVIDER selects the backend:
  'openai'    (default) — real AsyncOpenAI
  'anthropic'           — Anthropic SDK shim; structured output emulated via
                          forced tool-use (input_schema = Pydantic JSON schema)
  'gemini'              — google-genai SDK shim; structured output via native
                          response_schema=PydanticModel

All shims expose the same surface as AsyncOpenAI so callers can use
    .chat.completions.create(...)            (plain text)
    .beta.chat.completions.parse(            (Pydantic structured output)
        model=..., messages=..., response_format=PydanticModel, ...
    )
unchanged. response.choices[0].message.parsed returns a Pydantic instance.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any


def get_client() -> Any:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        return _ClaudeShim()
    if provider in ("google", "gemini"):
        return _GeminiShim()
    from openai import AsyncOpenAI
    return AsyncOpenAI()


class _ClaudeShim:
    def __init__(self) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()
        self.chat = SimpleNamespace(completions=_ClaudeChatCompletions(self._client))
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=_ClaudeBetaChatCompletions(self._client))
        )


def _split_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI lets system messages live in `messages`; Anthropic wants them
    as a top-level `system` parameter."""
    system_parts: list[str] = []
    user_messages: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    system = "\n\n".join(s for s in system_parts if s) or None
    return system, user_messages


def _usage(response: Any) -> SimpleNamespace:
    u = getattr(response, "usage", None)
    if u is None:
        return SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    return SimpleNamespace(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)


def _max_tokens(kwargs: dict) -> int:
    return int(
        kwargs.pop("max_completion_tokens", None)
        or kwargs.pop("max_tokens", None)
        or 4096
    )


class _ClaudeChatCompletions:
    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client

    async def create(self, *, model: str, messages: list[dict], **kwargs: Any) -> Any:
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("response_format", None)
        max_tokens = _max_tokens(kwargs)
        system, user_messages = _split_messages(messages)
        call_kwargs: dict[str, Any] = dict(model=model, messages=user_messages, max_tokens=max_tokens)
        if system is not None:
            call_kwargs["system"] = system
        for k in ("top_p", "stop_sequences"):
            if k in kwargs:
                call_kwargs[k] = kwargs.pop(k)
        response = await self._client.messages.create(**call_kwargs)
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text")
        message = SimpleNamespace(content=text, parsed=None, refusal=None)
        choice = SimpleNamespace(message=message, finish_reason=getattr(response, "stop_reason", "stop"), index=0)
        return SimpleNamespace(choices=[choice], usage=_usage(response), model=model)


class _ClaudeBetaChatCompletions:
    """Emulates `client.beta.chat.completions.parse(response_format=Model, ...)`
    via a forced tool_use call whose input_schema is the Pydantic JSON schema."""

    TOOL_NAME = "respond_with_schema"

    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client

    async def parse(self, *, model: str, messages: list[dict], response_format: Any, **kwargs: Any) -> Any:
        kwargs.pop("reasoning_effort", None)
        max_tokens = _max_tokens(kwargs)
        system, user_messages = _split_messages(messages)

        if hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
        else:
            schema = response_format.schema()
        schema = _flatten_refs(schema)

        tool = {
            "name": self.TOOL_NAME,
            "description": (
                f"Return your response by calling this tool. "
                f"The tool input must match the {response_format.__name__} schema."
            ),
            "input_schema": schema,
        }

        call_kwargs: dict[str, Any] = dict(
            model=model,
            messages=user_messages,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": self.TOOL_NAME},
        )
        if system is not None:
            call_kwargs["system"] = system

        response = await self._client.messages.create(**call_kwargs)

        tool_input: dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == self.TOOL_NAME:
                tool_input = getattr(block, "input", None)
                break
        if tool_input is None:
            raise RuntimeError(
                f"Claude did not return a tool_use block for {response_format.__name__}. "
                f"stop_reason={getattr(response, 'stop_reason', None)!r}"
            )

        if hasattr(response_format, "model_validate"):
            parsed = response_format.model_validate(tool_input)
        else:
            parsed = response_format.parse_obj(tool_input)

        message = SimpleNamespace(content=None, parsed=parsed, refusal=None)
        choice = SimpleNamespace(message=message, finish_reason=getattr(response, "stop_reason", "stop"), index=0)
        return SimpleNamespace(choices=[choice], usage=_usage(response), model=model)


def _flatten_refs(schema: dict) -> dict:
    """Inline `$defs` / `definitions` references so the schema can be passed
    to Anthropic as `input_schema`. Idempotent."""
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                for prefix in ("#/$defs/", "#/definitions/"):
                    if ref.startswith(prefix):
                        key = ref[len(prefix):]
                        if key in defs:
                            return resolve({k: v for k, v in defs[key].items()})
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


# ---------------------------------------------------------------------------
# Gemini (Google) shim
# ---------------------------------------------------------------------------

class _GeminiShim:
    """Duck-typed AsyncOpenAI replacement that routes to the Gemini API.

    Gemini has first-class structured output via `response_schema=PydanticModel`,
    so the parse() path is a thin re-shape of the response into the OpenAI
    surface (response.choices[0].message.parsed).
    """

    def __init__(self) -> None:
        from google import genai
        # Reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.
        self._client = genai.Client()
        self.chat = SimpleNamespace(completions=_GeminiChatCompletions(self._client))
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=_GeminiBetaChatCompletions(self._client))
        )


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI-style messages → (system_instruction, gemini_contents).

    Gemini has no `system` role inside `contents`; system text goes to
    `config.system_instruction`. Roles map: assistant→model, user→user.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
            continue
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(content)
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    system = "\n\n".join(s for s in system_parts if s) or None
    return system, contents


def _gemini_usage(response: Any) -> SimpleNamespace:
    u = getattr(response, "usage_metadata", None)
    if u is None:
        return SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    inp = getattr(u, "prompt_token_count", 0) or 0
    out = getattr(u, "candidates_token_count", 0) or 0
    return SimpleNamespace(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)


class _GeminiChatCompletions:
    def __init__(self, gemini_client: Any) -> None:
        self._client = gemini_client

    async def create(self, *, model: str, messages: list[dict], **kwargs: Any) -> Any:
        # OpenAI-only knobs we silently drop on the Gemini path.
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("response_format", None)
        max_tokens = _max_tokens(kwargs)
        system, contents = _to_gemini_contents(messages)

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system is not None:
            config["system_instruction"] = system
        for k in ("top_p", "temperature", "stop_sequences"):
            if k in kwargs:
                config[k] = kwargs.pop(k)

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", "") or ""
        message = SimpleNamespace(content=text, parsed=None, refusal=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        return SimpleNamespace(choices=[choice], usage=_gemini_usage(response), model=model)


class _GeminiBetaChatCompletions:
    """Emulates `client.beta.chat.completions.parse(response_format=Model, ...)`
    via Gemini's native `response_schema` structured-output mode."""

    def __init__(self, gemini_client: Any) -> None:
        self._client = gemini_client

    async def parse(self, *, model: str, messages: list[dict], response_format: Any, **kwargs: Any) -> Any:
        kwargs.pop("reasoning_effort", None)
        max_tokens = _max_tokens(kwargs)
        system, contents = _to_gemini_contents(messages)

        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": response_format,
        }
        if system is not None:
            config["system_instruction"] = system

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # google-genai returns response.parsed as a PydanticModel instance
        # when response_schema is a Pydantic class. Fall back to manual JSON
        # parsing for SDK versions or response shapes that don't populate it.
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            import json as _json
            text = getattr(response, "text", None) or ""
            try:
                data = _json.loads(text)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Gemini did not return parseable JSON for {response_format.__name__}: {exc}. "
                    f"Raw text head: {text[:200]!r}"
                )
            if hasattr(response_format, "model_validate"):
                parsed = response_format.model_validate(data)
            else:
                parsed = response_format.parse_obj(data)

        message = SimpleNamespace(content=None, parsed=parsed, refusal=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        return SimpleNamespace(choices=[choice], usage=_gemini_usage(response), model=model)