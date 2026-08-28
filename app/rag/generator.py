"""LLM provider abstraction.

``LLM_PROVIDER`` selects the back-end (``anthropic`` / ``openai`` / ``none``).
API keys are read from the environment only.  The interface is deliberately
minimal — a single "system + user -> JSON" call — because everything that makes
the answer trustworthy (retrieval, citations, validation, confidence) lives
outside the model.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Generation failed (network, auth, quota, malformed output)."""


class LLMNotConfigured(LLMError):
    """No provider/key configured — the app still works in retrieval-only mode."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(ABC):
    provider: str = "abstract"

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        ...

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Call the model and parse its JSON output."""
        response = self.complete(system, user)
        return parse_json_response(response.text)


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMNotConfigured("Пакет anthropic не установлен") from exc
        if not settings.anthropic_api_key:
            raise LLMNotConfigured("ANTHROPIC_API_KEY не задан в .env")
        options: dict[str, object] = {
            "api_key": settings.anthropic_api_key,
            "timeout": float(settings.llm_timeout_seconds),
        }
        if settings.anthropic_base_url:
            options["base_url"] = settings.anthropic_base_url
        self._client = anthropic.Anthropic(**options)  # type: ignore[arg-type]

    def complete(self, system: str, user: str) -> LLMResponse:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Anthropic request failed: %s", type(exc).__name__)
            raise LLMError(f"Ошибка обращения к Anthropic API: {exc}") from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            input_tokens=getattr(message.usage, "input_tokens", None),
            output_tokens=getattr(message.usage, "output_tokens", None),
        )


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMNotConfigured("Пакет openai не установлен") from exc
        if not settings.openai_api_key:
            raise LLMNotConfigured("OPENAI_API_KEY не задан в .env")
        options: dict[str, object] = {
            "api_key": settings.openai_api_key,
            "timeout": float(settings.llm_timeout_seconds),
        }
        if settings.openai_base_url:
            options["base_url"] = settings.openai_base_url
        self._client = OpenAI(**options)  # type: ignore[arg-type]

    def complete(self, system: str, user: str) -> LLMResponse:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI request failed: %s", type(exc).__name__)
            raise LLMError(f"Ошибка обращения к OpenAI API: {exc}") from exc

        usage = completion.usage
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model=self.model,
            provider=self.provider,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response that should be a single JSON object.

    Tolerates markdown fences and leading prose, because a strict failure here
    would discard an otherwise valid answer.
    """
    if not text or not text.strip():
        raise LLMError("Модель вернула пустой ответ")

    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Не удалось разобрать JSON из ответа модели: {exc}") from exc
    raise LLMError("В ответе модели не найден JSON-объект")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the configured client, raising :class:`LLMNotConfigured` if absent."""
    settings = settings or get_settings()
    model = settings.resolved_model_name()

    if settings.llm_provider == "anthropic":
        return AnthropicClient(model, settings)
    if settings.llm_provider == "openai":
        return OpenAIClient(model, settings)
    raise LLMNotConfigured(
        "LLM_PROVIDER=none — генерация отключена, доступен только режим поиска."
    )


def llm_available(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.llm_provider != "none" and settings.has_llm_credentials()
