from threading import Barrier

import pytest
from langchain_core.messages import AIMessage

from rag_langchain.council import ROLES, run_council


class CouncilChat:
    def __init__(self, failures=(), invalid=False, barrier=None):
        self.failures = failures
        self.invalid = invalid
        self.barrier = barrier
        self.calls = []

    def invoke(self, messages):
        system = messages[0].content
        self.calls.append(messages)
        role = next((r for r in ROLES if "Role: " + r in system), "Chair")
        if role != "Chair" and self.barrier:
            self.barrier.wait(timeout=10)
        if role in self.failures:
            raise RuntimeError("private provider diagnostic")
        return AIMessage(content="Alice follows a rabbit. " + ("[99]" if self.invalid else "[1]"))


def test_parallel_reviews_then_grounded_chair(setup):
    service, _, _, _ = setup
    service.ingest()
    chat = CouncilChat(barrier=Barrier(3))
    service._chat = chat
    progress = []
    answer = service.ask("What does Alice follow?", mode="council", progress=progress.append)
    assert answer.mode == "council"
    assert len(chat.calls) == 4
    assert len(answer.reviews) == 3
    assert all(r.status == "complete" for r in answer.reviews)
    assert [r.role for r in answer.reviews] == list(ROLES)
    assert answer.to_dict()["reviews"][0]["citations"] == [1]
    chair = chat.calls[-1]
    assert "council chair" in chair[0].content
    assert "Document excerpts:" in chair[1].content
    assert "Untrusted reviewer notes:" in chair[1].content
    assert progress[-1].startswith("Chair")


def test_partial_failure_is_visible(setup):
    service, _, _, _ = setup
    service.ingest()
    service._chat = CouncilChat(failures={"Skeptic"})
    answer = service.ask("Alice?", mode="council")
    assert answer.warnings
    assert answer.reviews[1].status == "failed"
    assert "private" not in str(answer.to_dict())
    assert answer.reviews[1].findings == ""


def test_council_requires_quorum_and_valid_citations():
    chat = CouncilChat(failures={"Skeptic", "Evidence analyst"})
    with pytest.raises(ValueError, match="fewer than two"):
        run_council("Alice?", "[1] Rabbit.", [1], lambda: chat)
    assert len(chat.calls) == 3  # No synthesis or silent fallback.
    with pytest.raises(ValueError, match="fewer than two"):
        run_council("Alice?", "[1] Rabbit.", [1], lambda: CouncilChat(invalid=True))


@pytest.mark.parametrize("content", ["Invented claim. [99]", "An unsupported answer."])
def test_chair_cannot_publish_missing_or_unknown_citations(content):
    class ChairChat(CouncilChat):
        def invoke(self, messages):
            if "council chair" in messages[0].content:
                return AIMessage(content=content)
            return super().invoke(messages)

    if "[99]" in content:
        with pytest.raises(ValueError, match="not supplied"):
            run_council("Alice?", "[1] Rabbit.", [1], ChairChat)
    else:
        answer, _, warnings = run_council("Alice?", "[1] Rabbit.", [1], ChairChat)
        assert "cannot establish" in answer
        assert warnings


def test_empty_context_avoids_all_model_calls(setup):
    service, _, chat, _ = setup
    service.ingest()
    service.search = lambda *args: []
    answer = service.ask("Alice?", mode="council")
    assert answer.mode == "council"
    assert not answer.reviews
    assert not chat.messages
    with pytest.raises(ValueError, match="mode"):
        service.ask("Alice?", mode="unknown")
