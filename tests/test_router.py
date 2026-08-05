from orchestra.config import load_agents
from orchestra.router import parse_triage_response, rank_agents, score_agent

AGENTS = load_agents()


def _top(task, text):
    return rank_agents(AGENTS, task, text)[0][0].name


def test_declared_task_type_wins():
    assert _top("review", "regarde ca") == "reviewer"
    assert _top("explain", "regarde ca") == "explainer"
    assert _top("test", "regarde ca") == "tester"
    assert _top("document", "regarde ca") == "documenter"


def test_partial_task_match():
    # "code-review" doit atteindre l'agent declarant "review".
    assert _top("code-review", "") == "reviewer"


def test_keywords_route_without_task_type():
    assert _top(None, "ecris les tests unitaires pytest de ce module") == "tester"
    assert _top(None, "resume ces logs de build") == "summarizer"


def test_multiword_keyword_survives_tokenisation():
    plain = score_agent(AGENTS["reviewer"], None, "fais une code review")
    assert plain > 0


def test_unrelated_text_scores_low():
    # Doit tomber sous le seuil d'ambiguite pour declencher le triage LLM.
    best_score = rank_agents(AGENTS, None, "bonjour, ca va ?")[0][1]
    assert best_score < 6.0


def test_triage_response_parsing():
    assert parse_triage_response('{"agent": "reviewer", "reason": "code"}', AGENTS) == (
        "reviewer",
        "code",
    )
    # Les petits modeles encadrent souvent le JSON de prose ou de fences.
    wrapped = 'Sure!\n```json\n{"agent": "tester", "reason": "tests"}\n```'
    assert parse_triage_response(wrapped, AGENTS)[0] == "tester"


def test_triage_hallucinated_agent_is_rejected():
    assert parse_triage_response('{"agent": "inexistant"}', AGENTS) is None
    assert parse_triage_response("pas du json", AGENTS) is None
