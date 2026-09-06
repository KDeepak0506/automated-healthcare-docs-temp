import json
import logging
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError

import groq
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(Exception):
    """Base exception for LLM service failures."""
    pass


class LLMConfigurationError(LLMServiceError):
    """Raised when LLM service is misconfigured (e.g., missing API key)."""
    pass


class LLMTimeoutError(LLMServiceError):
    """Raised when an external LLM request times out."""
    pass


class LLMResponseParsingError(LLMServiceError):
    """Raised when the LLM returns invalid JSON or fails schema validation."""
    pass


class LLMService:
    """
    Shared Groq LLM service for structured output generation.
    Enforces timeout, error translation, and privacy-safe logging.
    """

    def __init__(self, client: Groq | None = None) -> None:
        self._client = client

    def _get_client(self) -> Groq:
        if self._client is not None:
            return self._client

        api_key = settings.groq_api_key
        if not api_key or not api_key.strip():
            logger.error("Groq API key is missing or not configured")
            raise LLMConfigurationError("Groq API key is not configured")

        return Groq(
            api_key=api_key.strip(),
            timeout=settings.groq_timeout_seconds,
        )

    def _execute_completion(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, str] | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Internal completion executor handling client initialization,
        reasoning effort configuration, timeouts, and exception wrapping.
        """
        client = self._get_client()

        logger.info(
            f"Dispatching completion to Groq model={settings.groq_model} "
            f"timeout={settings.groq_timeout_seconds}s"
        )

        completion_kwargs: dict[str, Any] = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            completion_kwargs["response_format"] = response_format

        # gpt-oss models support reasoning_effort to cap hidden chain-of-thought
        # token usage. Low effort keeps free-tier TPM consumption reasonable for
        # short structured-output tasks like classification/summarization,
        # which don't need deep multi-step reasoning.
        if "gpt-oss" in settings.groq_model:
            completion_kwargs["reasoning_effort"] = "low"

        try:
            chat_completion = client.chat.completions.create(**completion_kwargs)
        except APITimeoutError as exc:
            logger.error(f"Groq API request timed out after {settings.groq_timeout_seconds}s")
            raise LLMTimeoutError("Groq request timed out") from exc
        except APIConnectionError as exc:
            logger.error("Groq API network connection failed")
            raise LLMServiceError("Groq connection error") from exc
        except APIStatusError as exc:
            logger.error(f"Groq API returned HTTP status {exc.status_code}")
            raise LLMServiceError(f"Groq API error: status {exc.status_code}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error during Groq LLM execution: {exc}")
            raise LLMServiceError("Failed to communicate with LLM provider") from exc

        choices = chat_completion.choices
        if not choices or not choices[0].message or not choices[0].message.content:
            logger.error("Groq returned empty response choices or content")
            raise LLMResponseParsingError("Groq returned an empty response")

        return choices[0].message.content.strip()

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate free-form text response (e.g., for RAG Q&A).
        Only sanitized text must be provided in prompts.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._execute_completion(
            messages=messages,
            response_format=None,
            temperature=temperature,
        )

    def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """
        Generate structured JSON output validated against a Pydantic schema.
        Only sanitized text must be provided in prompts.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        raw_content = self._execute_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            parsed_json = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from Groq response")
            raise LLMResponseParsingError("LLM response is not valid JSON") from exc

        try:
            validated_result = response_schema.model_validate(parsed_json)
        except ValidationError as exc:
            logger.error(f"Schema validation failed for LLM response against {response_schema.__name__}")
            raise LLMResponseParsingError(f"LLM output failed schema validation: {exc}") from exc

        logger.info(f"Successfully validated structured response for {response_schema.__name__}")
        return validated_result



# Singleton instance for shared usage
llm_service = LLMService()
