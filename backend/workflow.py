"""LangGraph workflow for routing, tool execution, and response composition."""
from __future__ import annotations

from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # Keep the app usable before optional dependencies are installed.
    END = START = StateGraph = None


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    session_context: dict
    routing: dict
    intent: str
    entities: dict
    plan: dict
    tool_calls: list[dict]
    result: dict
    response: dict


def build_workflow(orchestrator) -> Any:
    """Build a compiled workflow using the existing router and tool registry."""
    if StateGraph is None:
        return _FallbackWorkflow(orchestrator)

    graph = StateGraph(AgentState)

    def route(state: AgentState) -> dict:
        routing = orchestrator.router.classify(state['user_query'], state['session_context'])
        intent = routing['intent']
        entities = routing.get('entities') or {}
        plan = routing.get('plan') or {
            'primary_action': intent,
            'tools': ['instrument_search'],
            'needs_clarification': False,
            'entities': entities,
        }
        tool_calls = orchestrator.tools.build_tool_calls(
            state['user_query'], intent, entities, state['session_context'],
        )
        return {
            'routing': routing,
            'intent': intent,
            'entities': entities,
            'plan': plan,
            'tool_calls': tool_calls,
        }

    def execute_tool(state: AgentState) -> dict:
        tool_call = state['tool_calls'][0]
        raw_result = orchestrator.tools.execute(
            tool_call['tool'],
            query=state['user_query'],
            entities=state['entities'],
            session_context=state['session_context'],
            _intent=state['intent'],
        )
        return {
            'result': orchestrator.normalize_tool_result(
                tool_call['tool'], state['intent'], raw_result, state['user_query'],
            ),
        }

    def compose(state: AgentState) -> dict:
        response = dict(state['result'])
        response['routing'] = state['routing']
        response['plan'] = state['plan']
        response['tool_calls'] = state['tool_calls']
        response['tool_summary'] = {
            'called': state['tool_calls'][0]['tool'],
            'intent': state['intent'],
            'query': state['user_query'],
        }
        response['text'] = orchestrator._summarize_response(
            state['user_query'], state['intent'], response,
        )
        response.setdefault('intent', state['intent'])
        return {'response': response}

    graph.add_node('route', route)
    graph.add_node('execute_tool', execute_tool)
    graph.add_node('compose', compose)
    graph.add_edge(START, 'route')
    graph.add_edge('route', 'execute_tool')
    graph.add_edge('execute_tool', 'compose')
    graph.add_edge('compose', END)
    return graph.compile()


class _FallbackWorkflow:
    """Minimal compatibility runner for environments without LangGraph installed."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def invoke(self, state: AgentState) -> AgentState:
        routing = self.orchestrator.router.classify(state['user_query'], state['session_context'])
        intent = routing['intent']
        entities = routing.get('entities') or {}
        plan = routing.get('plan') or {
            'primary_action': intent,
            'tools': ['instrument_search'],
            'needs_clarification': False,
            'entities': entities,
        }
        tool_calls = self.orchestrator.tools.build_tool_calls(
            state['user_query'], intent, entities, state['session_context'],
        )
        tool_call = tool_calls[0]
        raw_result = self.orchestrator.tools.execute(
            tool_call['tool'], query=state['user_query'], entities=entities,
            session_context=state['session_context'],
            _intent=intent,
        )
        result = self.orchestrator.normalize_tool_result(
            tool_call['tool'], intent, raw_result, state['user_query'],
        )
        response = dict(result)
        response.update({
            'routing': routing,
            'plan': plan,
            'tool_calls': tool_calls,
            'tool_summary': {
                'called': tool_call['tool'], 'intent': intent, 'query': state['user_query'],
            },
        })
        response['text'] = self.orchestrator._summarize_response(
            state['user_query'], intent, response,
        )
        response.setdefault('intent', intent)
        return {'response': response}