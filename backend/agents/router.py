"""Query intent router — classifies user messages into agent branches."""
from __future__ import annotations

import logging
import re

from backend.services.llm_client import LLMClient
from backend.utils import sanitize_input
from config import Config

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset({
    'find_instrument',
    'instrument_details',
    'compare',
    'why',
    'handbook',
})

ACTION_TOOLS = {
    'find_instrument': ['instrument_search'],
    'instrument_details': ['project_usage_lookup'],
    'compare': ['compare_instruments'],
    'why': ['instrument_search'],
    'handbook': ['handbook_lookup'],
}

PREVIOUS_USAGE_TOOL = 'previous_usage_lookup'

WHY_PATTERNS = (
    r'\bwhy\b', r'\bhow come\b', r'为什么', r'為什麼', r'原因', r'explain.*recommend',
    r'why.*(first|top|rank|recommend)', r'为什么.*(推荐|推薦|排)',
)

PROJECT_NO_PATTERN = re.compile(r'\bP\d{4}-\d{3}\b', re.IGNORECASE)


def _regex_why(query_l: str, session_context: dict) -> bool:
    if not session_context.get('last_matched'):
        return False
    return any(re.search(p, query_l) for p in WHY_PATTERNS)


def _extract_project_no(text: str) -> str | None:
    m = PROJECT_NO_PATTERN.search(text)
    return m.group(0).upper() if m else None


class Router:
    def __init__(self, llm: LLMClient | None = None, config: type[Config] = Config):
        self.llm = llm or LLMClient(config)
        self.config = config

    def _make_plan(self, intent: str, entities: dict | None = None, *, needs_clarification: bool = False) -> dict:
        return {
            'primary_action': intent,
            'tools': ACTION_TOOLS.get(intent, ['instrument_search']),
            'needs_clarification': needs_clarification,
            'entities': entities or {'instruments': [], 'project_no': None, 'compare_targets': []},
        }

    def plan_query(self, query: str, session_context: dict | None = None) -> dict:
        session_context = session_context or {}
        query = sanitize_input(query, max_length=self.config.MAX_QUERY_LENGTH)
        query_l = query.lower()

        if not query.strip():
            return {
                'intent': 'handbook',
                'entities': {'instruments': [], 'project_no': None, 'compare_targets': []},
                'confidence': 0.4,
                'plan': self._make_plan('handbook', needs_clarification=True),
            }

        if _regex_why(query_l, session_context):
            return {
                'intent': 'why',
                'entities': {'instruments': [], 'project_no': None, 'compare_targets': []},
                'confidence': 0.95,
                'plan': self._make_plan('why'),
            }

        previous_usage_patterns = (
            'used before', 'used previously', 'used in previous project', 'previous project used',
            'any previous project', 'has this been used before', 'was it used before',
            'used in project', 'project used', 'previously used', 'ever used', 'has been used',
            'has it been used', 'been used before', 'previously used this', 'previous work',
            'prior work', 'past work', 'in previous work', 'any use of', 'used in previous work',
            'used in past work', 'history of use', 'ever used in', 'has phq', 'has gad'
        )
        previous_usage_re = re.search(
            r'\b(phq(?:[- ]?\d+)?|gad(?:[- ]?\d+)?|who(?:[- ]?\d+)?|lawton|instrument|scale|questionnaire)\b',
            query_l,
        )
        asks_about_previous_use = (
            any(w in query_l for w in previous_usage_patterns)
            or re.search(r'\b(previous|prior|past)\b.*\b(work|project|study)\b', query_l)
            or ('before' in query_l and re.search(r'\b(used|use|usage)\b', query_l))
        )
        if asks_about_previous_use and previous_usage_re:
            plan = self._make_plan('instrument_details', {'instruments': [], 'project_no': None, 'compare_targets': []})
            plan['tools'] = [PREVIOUS_USAGE_TOOL]
            return {
                'intent': 'instrument_details',
                'entities': {'instruments': [], 'project_no': None, 'compare_targets': []},
                'confidence': 0.92,
                'plan': plan,
            }

        proj_no = _extract_project_no(query)
        if proj_no and any(w in query_l for w in ('used', 'usage', 'project', '用了', '项目', '項目')):
            return {
                'intent': 'instrument_details',
                'entities': {'instruments': [], 'project_no': proj_no, 'compare_targets': []},
                'confidence': 0.9,
                'plan': self._make_plan('instrument_details', {'instruments': [], 'project_no': proj_no, 'compare_targets': []}),
            }

        system = (
            'You are an intent classifier for a measurement instrument assistant.\n'
            'Classify the user query into exactly ONE intent.\n'
            'Intent definitions:\n'
            '- find_instrument: search or recommend instruments for a need\n'
            '- instrument_details: lookup which instruments a project uses, or usage of a specific instrument in projects\n'
            '- compare: compare two or more named instruments\n'
            '- why: explain previous recommendations (only when user asks why/how come about prior results)\n'
            '- handbook: general usage questions, what the system can do, definitions\n\n'
            'Important: if the user asks whether an instrument was used before, or whether a previous project used an instrument, classify it as instrument_details and set tool intent to previous_usage_lookup when applicable.\n'
            'The project usage workbook uses these columns: Measurement Instrument, Acronym, Outcome Domain, Actual Use in Project.\n'
            'Respond with JSON only:\n'
            '{"intent":"...", "entities":{"instruments":[],"project_no":null,"compare_targets":[]}, "confidence":0.0}\n'
            'Extract instrument names/acronyms into instruments or compare_targets as appropriate.\n'
            'Extract project numbers like P2024-001 into project_no.'
        )

        history_hint = ''
        if session_context.get('last_intent'):
            history_hint = (
                f"\nPrevious turn: intent={session_context.get('last_intent')}, "
                f"query={session_context.get('last_query', '')[:120]}"
            )

        user = f'User query: {query}{history_hint}'

        try:
            parsed = self.llm.chat_json(system, user, model=self.config.ROUTER_MODEL, max_tokens=200)
            if isinstance(parsed, dict):
                intent = parsed.get('intent', 'handbook')
                if intent not in VALID_INTENTS:
                    intent = 'handbook'
                entities = parsed.get('entities') or {}
                routing = {
                    'intent': intent,
                    'entities': {
                        'instruments': entities.get('instruments') or [],
                        'project_no': entities.get('project_no') or _extract_project_no(query),
                        'compare_targets': entities.get('compare_targets') or [],
                    },
                    'confidence': float(parsed.get('confidence', 0.8)),
                }
                routing['plan'] = self._make_plan(intent, routing['entities'])
                return routing
        except Exception as e:
            logger.warning('Router LLM failed, falling back to heuristics: %s', e)

        if any(w in query_l for w in ('compare', 'vs', 'versus', 'difference', '比较', '對比', '对比')):
            return {
                'intent': 'compare',
                'entities': {'instruments': [], 'project_no': None, 'compare_targets': []},
                'confidence': 0.6,
                'plan': self._make_plan('compare'),
            }

        return {
            'intent': 'find_instrument',
            'entities': {'instruments': [], 'project_no': None, 'compare_targets': []},
            'confidence': 0.5,
            'plan': self._make_plan('find_instrument'),
        }

    def classify(self, query: str, session_context: dict | None = None) -> dict:
        return self.plan_query(query, session_context)
