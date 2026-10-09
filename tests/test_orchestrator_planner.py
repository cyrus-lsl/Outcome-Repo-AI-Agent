from backend.agents.router import Router
from backend.orchestrator import Orchestrator


def test_router_plan_query_returns_tool_plan():
    plan = Router().plan_query('Compare PHQ-9 and GAD-7', {})

    assert plan['primary_action'] == 'compare'
    assert 'compare_instruments' in plan['tools']
    assert plan['needs_clarification'] is False


def test_orchestrator_adds_structured_plan_to_response():
    response = Orchestrator().handle_message('demo-session', 'Compare PHQ-9 and GAD-7')

    assert response['intent'] == 'compare'
    assert response['plan']['primary_action'] == 'compare'
    assert 'compare_instruments' in response['plan']['tools']
    assert response['status'] == 'ok'
    assert 'compare' in response['text'].lower() or 'compared' in response['text'].lower()


def test_orchestrator_keeps_detailed_why_explanation():
    orchestrator = Orchestrator()
    response = {
        'text': 'Lawton is useful because it measures ability to manage daily tasks and function in older adults.',
        'matched': [{'name': 'Lawton Instrumental Activities of Daily Living'}],
    }

    summary = orchestrator._summarize_response(
        'why should I use Lawton Instrumental Activities of Daily Living',
        'why',
        response,
    )

    assert 'Lawton' in summary
    assert 'daily tasks' in summary.lower()
    assert 'prioritized the strongest fit' not in summary.lower()


def test_router_routes_previous_project_usage_query_to_project_lookup():
    plan = Router().plan_query('any previous project used PHQ-9 before', {})

    assert plan['primary_action'] == 'instrument_details'
    assert plan['tools'] == ['previous_usage_lookup']
