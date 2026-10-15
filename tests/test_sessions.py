"""Cloud sessions the gym creates are training sessions that need no viewer."""

from openbeam_gym import vector
from openbeam_gym.mcp import CloudAccount, ToolResult


class RecordingAccount(CloudAccount):
    """A CloudAccount that records its tool calls instead of reaching the cloud."""

    def __init__(self):
        self.base_url = "https://cloud.example.test"
        self.calls = []

    def call(self, tool, arguments=None, raise_on_error=True, timeout=None):
        self.calls.append((tool, arguments or {}))
        if tool == "cloud_session_start":
            n = sum(1 for t, _ in self.calls if t == tool)
            return ToolResult({"session_id": f"trn_{n:024d}", "state": "creating", "player_url": "https://x.test/p#t",
                               "kind": "training"})
        return ToolResult({})


def test_start_session_asks_for_a_training_session():
    account = RecordingAccount()
    started = account.start_session()
    assert account.calls == [("cloud_session_start", {"width": 1280, "height": 720, "fps": 60, "kind": "training"})]
    assert started["kind"] == "training"


def test_vector_sessions_are_training_and_open_no_browser(monkeypatch):
    opened, ready = [], []
    monkeypatch.setattr(vector.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(vector.CloudGame, "wait_ready", lambda self, timeout_s=300.0: ready.append(self.session_id))
    account = RecordingAccount()
    ids = vector.start_cloud_sessions(account, 2)
    assert len(ids) == 2 and all(i.startswith("trn_") for i in ids)
    assert [a["kind"] for t, a in account.calls if t == "cloud_session_start"] == ["training", "training"]
    assert opened == [] and ready == ids
