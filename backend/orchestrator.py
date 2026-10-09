"""Central orchestrator — route queries to agent branches and update session."""
from __future__ import annotations

import logging

from backend.agents.compare import CompareAgent
from backend.agents.find_instrument import FindInstrumentAgent
from backend.agents.handbook import HandbookAgent
from backend.agents.instrument_details import InstrumentDetailsAgent
from backend.agents.previous_usage import PreviousUsageAgent
from backend.agents.router import Router
from backend.services.session_store import store
from backend.utils import sanitize_input
from backend.workflow import build_workflow
from config import Config

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, config: type[Config] = Config):
        self.config = config
        self.agents = {
            'find_instrument': FindInstrumentAgent(config=config),
            'instrument_details': InstrumentDetailsAgent(),
            'previous_usage': PreviousUsageAgent(),
            'compare': CompareAgent(config=config),
            'why': FindInstrumentAgent(config=config),
            'handbook': HandbookAgent(config=config),
        }

    def build_tool_calls(self, user_query: str, intent: str, entities: dict, session_context: dict | None = None) -> list[dict]:
        tool_map = {
            'find_instrument': {'name': 'instrument_search', 'args': {'query': user_query, 'entities': entities}},
            'instrument_details': {'name': 'project_usage_lookup', 'args': {'query': user_query, 'entities': entities}},
            'compare': {'name': 'compare_instruments', 'args': {'query': user_query, 'entities': entities}},
            'why': {'name': 'explain_last_recommendations', 'args': {'query': user_query, 'entities': entities, 'session_context': session_context or {}}},
            'handbook': {'name': 'handbook_lookup', 'args': {'query': user_query, 'entities': entities}},
        }

        if intent == 'instrument_details' and any(token in user_query.lower() for token in ('previous', 'before', 'ever used', 'used before', 'used previously', 'has been used')):
            tool_map['instrument_details'] = {'name': 'previous_usage_lookup', 'args': {'query': user_query, 'entities': entities}}
        tool = tool_map.get(intent, tool_map['handbook'])
        return [{
            'tool': tool['name'],
            'args': tool['args'],
            'intent': intent,
        }]

    def execute(self, tool_name: str, **kwargs) -> dict:
        intent = kwargs.pop('_intent', None) or {
            'instrument_search': 'find_instrument',
            'project_usage_lookup': 'instrument_details',
            'previous_usage_lookup': 'previous_usage',
            'compare_instruments': 'compare',
            'explain_last_recommendations': 'why',
            'handbook_lookup': 'handbook',
        }.get(tool_name, 'handbook')

        agent = self.agents.get(intent)
        if agent is None:
            raise KeyError(f'Unknown tool: {tool_name}')

        if intent == 'why':
            return agent.run(
                kwargs.get('query', ''), kwargs.get('entities') or {},
                session_context=kwargs.get('session_context') or {},
            )
        if intent == 'compare':
            return agent.run(kwargs.get('query', ''), kwargs.get('entities') or {}, kwargs.get('session_context') or {})
        if intent == 'previous_usage':
            return agent.run(kwargs.get('query', ''), kwargs.get('entities') or {})
        return agent.run(kwargs.get('query', ''), kwargs.get('entities') or {})


class Orchestrator:
    def __init__(self, config: type[Config] = Config):
        self.config = config
        self.router = Router(config=config)
        self.tools = ToolRegistry(config=config)
        self.workflow = build_workflow(self)

    def _summarize_response(self, user_query: str, intent: str, result: dict) -> str:
        base_text = str(result.get('text') or '').strip()
        matched = result.get('matched') or []

        if intent == 'compare' and matched:
            names = [m.get('name') or 'Instrument' for m in matched[:4] if m.get('name')]
            return (
                f"I compared {', '.join(names)} for your question about \"{user_query}\".\n\n"
                + '\n\n'.join(
                    f"- {m.get('name')}: {m.get('purpose') or 'Purpose not specified'}; "
                    f"target: {m.get('target') or 'not specified'}; "
                    f"HK validated: {m.get('validated') or 'not specified'}"
                    for m in matched[:3]
                )
            )

        if intent == 'find_instrument' and matched:
            top = matched[:3]
            lines = [
                f"I found {len(matched)} instruments relevant to \"{user_query}\". The strongest matches are:"
            ]
            for idx, item in enumerate(top, 1):
                lines.append(
                    f"{idx}. {item.get('name')} — {item.get('purpose') or 'purpose not specified'}"
                )
            return '\n'.join(lines)

        if intent == 'instrument_details' and result.get('details'):
            return (
                f"I checked the project usage records related to \"{user_query}\" and found "
                f"{len(result['details'])} record(s).\n\n{base_text}"
            )

        if intent == 'why':
            if base_text:
                return base_text
            if matched:
                reasons = ', '.join(item.get('name', 'instrument') for item in matched[:3])
                return (
                    f"The earlier recommendation prioritized the strongest fit for your query, especially {reasons}. "
                    f"These items were ranked based on purpose alignment, target group fit, domain match, and validation status."
                )

        return base_text or 'I can help with that.'

    def normalize_tool_result(self, tool_name: str, intent: str, raw_result: dict, user_query: str) -> dict:
        normalized = dict(raw_result)
        normalized.setdefault('intent', intent)
        normalized.setdefault('tool', tool_name)
        normalized.setdefault('status', 'ok' if (raw_result.get('text') or raw_result.get('matched') or raw_result.get('details')) else 'empty')
        normalized.setdefault('raw_text', raw_result.get('text'))
        normalized.setdefault('matched', raw_result.get('matched') or [])
        normalized.setdefault('details', raw_result.get('details') or [])
        normalized.setdefault('evidence', {'tool': tool_name, 'query': user_query})
        normalized.setdefault('needs_clarification', False)
        return normalized

    def handle_message(self, session_id: str, user_query: str) -> dict:
        store.ensure_session(session_id)
        user_query = sanitize_input(user_query, max_length=self.config.MAX_QUERY_LENGTH)
        session_context = store.get_context(session_id)

        state = self.workflow.invoke({
            'session_id': session_id,
            'user_query': user_query,
            'session_context': session_context,
        })
        response = state['response']
        logger.info('Completed workflow intent=%s session=%s', response.get('intent'), session_id)

        store.update(session_id, response['intent'], user_query, response)
        return response


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
