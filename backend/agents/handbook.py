"""Handbook agent — answer general / usage questions from user guidelines."""
from __future__ import annotations

import logging

from backend.services.data_loader import load_handbook
from backend.services.llm_client import LLMClient
from backend.utils import sanitize_input
from config import Config

logger = logging.getLogger(__name__)


class HandbookAgent:
    def __init__(self, llm: LLMClient | None = None, config: type[Config] = Config):
        self.llm = llm or LLMClient(config)
        self.config = config

    def run(self, query: str, entities: dict | None = None) -> dict:
        del entities
        query = sanitize_input(query)
        try:
            handbook = load_handbook()
        except FileNotFoundError:
            handbook = 'No handbook file found. The assistant can search instruments, compare tools, and look up project usage.'

        system = (
            'You are a helpful guide for the Measurement Instrument Assistant.\n'
            'Answer ONLY based on the USER GUIDELINES below.\n'
            'If the question is about finding or comparing specific instruments, '
            'tell the user to ask a direct search or comparison question instead.\n'
            'Be concise and friendly. Use markdown when helpful.\n\n'
            f'USER GUIDELINES:\n{handbook}'
        )
        user = f'User question: {query}'

        try:
            text = self.llm.chat(system, user, max_tokens=600)
        except Exception as e:
            logger.error('Handbook LLM failed: %s', e, exc_info=True)
            text = (
                'I can help you **find instruments**, **look up project usage**, '
                '**compare tools**, and **explain recommendations**.\n\n'
                'Try: *"mental health for elderly"* or *"What did P2024-001 use?"*'
            )

        return {
            'intent': 'handbook',
            'text': text,
            'matched': [],
        }
