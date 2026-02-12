"""OpenAI API wrapper for chat completions and embeddings."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _get_openai_client():
    try:
        from openai import OpenAI
        if not settings.openai_api_key:
            logger.warning("OpenAI API key not configured – LLM calls will fail")
            return None
        return OpenAI(api_key=settings.openai_api_key)
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None


class LLMClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self):
        self.client = _get_openai_client()
        self.model = settings.openai_model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialised (missing API key or package)")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict:
        """Request structured JSON output from the model."""
        raw = self.chat(
            system_prompt,
            user_prompt,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)


class EmbeddingClient:
    """Thin wrapper around OpenAI embeddings."""

    def __init__(self):
        self.client = _get_openai_client()
        self.model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialised")
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]

    def embed_single(self, text: str) -> list[float]:
        results = self.embed([text])
        return results[0] if results else []
