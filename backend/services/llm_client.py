"""OpenAI-compatible LLM client wrapper."""
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: type[Config] = Config):
        self.config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = self.config.LLM_API_KEY or 'not-set'
            self._client = OpenAI(
                base_url=self.config.LLM_BASE_URL,
                api_key=api_key,
            )
        return self._client

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        completion = self._get_client().chat.completions.create(
            model=model or self.config.LLM_MODEL,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ''

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 400,
    ) -> dict | list:
        raw = self.chat(system, user, model=model, max_tokens=max_tokens)
        return parse_json_response(raw)


def parse_json_response(raw: str) -> dict | list:
    text = raw.strip()
    if '```' in text:
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if m := re.search(r'[\[{][\s\S]*[\]}]', text):
            return json.loads(m.group())
        raise
