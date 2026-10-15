"""MCP clients for an OpenBeam game.

Two ways in, one interface (`GameClient.call(tool, arguments)`):

* `LocalGame` - a game started with `ROR_AGENT_PORT` serves MCP itself at
  `POST http://host:port/mcp` with a bearer token. The endpoint file the game
  writes (`ROR_AGENT_ENDPOINT_FILE`, or `<user dir>/agent-endpoint.json`)
  names the host, port and token.
* `CloudGame` - one session of the OpenBeam cloud, reached through the
  account endpoint `https://cloud.openbeam.world/mcp` (OAuth, see oauth.py)
  and its `cloud_game_call` tool, which forwards to the game in the session.

Both speak MCP Streamable HTTP with JSON responses (protocol 2025-06-18), one
request per HTTP connection-reuse session (urllib keeps it simple; the game
closes every connection anyway).
"""

from __future__ import annotations

import base64
import json
import threading
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class McpError(RuntimeError):
    """A JSON-RPC error, or a transport failure."""


class ToolError(RuntimeError):
    """A tool answered with isError: the game refused the request."""

    def __init__(self, tool: str, message: str, body: Any = None):
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.body = body


@dataclass
class ToolResult:
    data: dict            # structuredContent, or the text parsed as JSON
    images: list = field(default_factory=list)  # [(mime, bytes)]
    text: str = ""
    seconds: float = 0.0  # round trip


def _decode_tool_result(tool: str, result: dict, seconds: float, raise_on_error: bool) -> ToolResult:
    texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    images = [(c.get("mimeType", ""), base64.b64decode(c.get("data", "")))
              for c in result.get("content", []) if c.get("type") == "image"]
    data = result.get("structuredContent")
    if data is None:
        try:
            data = json.loads(texts[0]) if texts else {}
        except ValueError:
            data = {"text": texts[0] if texts else ""}
    if raise_on_error and result.get("isError"):
        message = data.get("message") or data.get("error") or (texts[0] if texts else "tool error")
        if isinstance(message, dict):
            message = json.dumps(message)
        raise ToolError(tool, str(message), data)
    return ToolResult(data=data if isinstance(data, dict) else {"value": data}, images=images,
                      text=texts[0] if texts else "", seconds=seconds)


class HttpMcp:
    """A minimal MCP Streamable HTTP client (JSON responses, SSE tolerated)."""

    def __init__(self, url: str, token_provider, timeout: float = 120.0, user_agent: str = "openbeam-gym"):
        self.url = url
        self._token = token_provider
        self.timeout = timeout
        self.user_agent = user_agent
        self._id = 0
        self._lock = threading.Lock()
        self.session_id: str | None = None
        self.initialized = False
        self._parsed = urllib.parse.urlparse(url)
        self._conn = None
        self._io_lock = threading.Lock()

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def rpc(self, method: str, params: dict | None = None, notification: bool = False, timeout: float | None = None):
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            body["id"] = self._next_id()
        if params is not None:
            body["params"] = params
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION, "User-Agent": self.user_agent}
        token = self._token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        raw = self._post(json.dumps(body).encode(), headers, timeout or self.timeout, method)
        if notification or not raw.strip():
            return None
        answer = None
        if raw.lstrip().startswith("{"):
            answer = json.loads(raw)
        else:  # text/event-stream: the last data block carries the response
            for block in raw.split("\n\n"):
                data = "\n".join(l[5:].strip() for l in block.splitlines() if l.startswith("data:"))
                if data:
                    answer = json.loads(data)
        if answer is None:
            raise McpError(f"{method}: empty answer")
        if "error" in answer:
            raise McpError(f"{method}: {answer['error'].get('message')}")
        return answer.get("result")

    def _connection(self, timeout: float):
        if self._conn is None:
            cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
            self._conn = cls(self._parsed.hostname, self._parsed.port, timeout=timeout)
        self._conn.timeout = timeout
        if self._conn.sock is not None:
            self._conn.sock.settimeout(timeout)
        return self._conn

    def _post(self, data: bytes, headers: dict, timeout: float, method: str) -> str:
        """POST over one persistent connection (the game keeps POST /mcp
        connections open); reconnect once if the server closed it."""
        path = self._parsed.path or "/"
        with self._io_lock:
            for attempt in (0, 1):
                reused = self._conn is not None
                conn = self._connection(timeout)
                try:
                    conn.request("POST", path, body=data, headers=headers)
                    response = conn.getresponse()
                    raw = response.read().decode("utf-8", "replace")
                except (http.client.HTTPException, ConnectionError, BrokenPipeError, OSError) as e:
                    conn.close()
                    self._conn = None
                    # Retry only a reused connection the server had closed
                    # while idle; a fresh one failing is a real failure, and
                    # a tool call must not run twice.
                    if attempt == 0 and reused and not isinstance(e, TimeoutError):
                        continue
                    raise McpError(f"{method}: {e}") from None
                sid = response.getheader("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                if (response.getheader("Connection") or "").lower() == "close":
                    conn.close()
                    self._conn = None
                if response.status >= 400:
                    raise McpError(f"{method}: HTTP {response.status} {raw[:500]}")
                return raw
        raise McpError(f"{method}: no answer")

    def initialize(self, client_name: str = "openbeam-gym") -> dict:
        result = self.rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                         "clientInfo": {"name": client_name, "version": "0.1"}})
        self.rpc("notifications/initialized", notification=True)
        self.initialized = True
        return result or {}


class GameClient:
    """What the environment needs from a game: call a game tool."""

    def call(self, tool: str, arguments: dict | None = None, raise_on_error: bool = True,
             timeout: float | None = None) -> ToolResult:
        raise NotImplementedError

    def tools(self) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalGame(GameClient):
    """A game on this machine (or any host) serving POST /mcp."""

    def __init__(self, host: str, port: int, token: str, timeout: float = 120.0):
        self.host, self.port, self._token = host, int(port), token
        self.mcp = HttpMcp(f"http://{host}:{port}/mcp", lambda: self._token, timeout=timeout)
        self.mcp.initialize()

    @classmethod
    def from_endpoint_file(cls, path: str | Path, wait_s: float = 0.0, **kw) -> "LocalGame":
        path = Path(path).expanduser()
        deadline = time.time() + wait_s
        while True:
            try:
                info = json.loads(path.read_text())
                return cls(info.get("host", "127.0.0.1"), info["port"], info["token"], **kw)
            except (OSError, ValueError, KeyError, McpError):
                if time.time() >= deadline:
                    raise
                time.sleep(0.5)

    def call(self, tool, arguments=None, raise_on_error=True, timeout=None) -> ToolResult:
        started = time.perf_counter()
        result = self.mcp.rpc("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=timeout)
        return _decode_tool_result(tool, result or {}, time.perf_counter() - started, raise_on_error)

    def tools(self) -> list[dict]:
        return (self.mcp.rpc("tools/list", {}) or {}).get("tools", [])


class SessionGame(GameClient):
    """One cloud session's game through its own MCP endpoint,
    `{base_url}/v1/sessions/{session_id}/mcp`, with a bearer token that may
    use it (the account's MCP access token, or a service credential)."""

    def __init__(self, base_url: str, session_id: str, token_provider, timeout: float = 120.0):
        self.session_id = session_id
        self.mcp = HttpMcp(f"{base_url.rstrip('/')}/v1/sessions/{session_id}/mcp", token_provider, timeout=timeout)

    def call(self, tool, arguments=None, raise_on_error=True, timeout=None) -> ToolResult:
        started = time.perf_counter()
        result = self.mcp.rpc("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=timeout)
        return _decode_tool_result(tool, result or {}, time.perf_counter() - started, raise_on_error)

    def tools(self) -> list[dict]:
        return (self.mcp.rpc("tools/list", {}) or {}).get("tools", [])


class CloudAccount:
    """The account endpoint: cloud_session_* and cloud_game_call."""

    def __init__(self, base_url: str = "https://cloud.openbeam.world", token_provider=None, timeout: float = 180.0):
        from .oauth import OAuthTokens
        self.base_url = base_url.rstrip("/")
        self.auth = token_provider or OAuthTokens(self.base_url)
        self.mcp = HttpMcp(f"{self.base_url}/mcp", self.auth.access_token, timeout=timeout)
        self.mcp.initialize()

    def call(self, tool: str, arguments: dict | None = None, raise_on_error: bool = True,
             timeout: float | None = None) -> ToolResult:
        started = time.perf_counter()
        result = self.mcp.rpc("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=timeout)
        return _decode_tool_result(tool, result or {}, time.perf_counter() - started, raise_on_error)

    def sessions(self) -> list[dict]:
        return self.call("cloud_session_list").data.get("sessions", [])

    def start_session(self, width: int = 1280, height: int = 720, fps: int = 60, kind: str = "training") -> dict:
        """Start a cloud session. `kind="training"` (the default here) is a
        robot-learning session: it may use stepped physics and runs without a
        player tab, ending after 10 minutes without an MCP request or 2 hours
        without a viewer (or the account's normal session limit). `"play"`
        needs its player_url open, like the Play button."""
        return self.call("cloud_session_start", {"width": width, "height": height, "fps": fps, "kind": kind}).data

    def end_session(self, session_id: str) -> dict:
        return self.call("cloud_session_end", {"session_id": session_id}).data


class CloudGame(GameClient):
    """One cloud session's game, through cloud_game_call."""

    def __init__(self, account: CloudAccount, session_id: str):
        self.account = account
        self.session_id = session_id

    def call(self, tool, arguments=None, raise_on_error=True, timeout=None) -> ToolResult:
        started = time.perf_counter()
        result = self.account.mcp.rpc("tools/call", {"name": "cloud_game_call", "arguments": {
            "session_id": self.session_id, "name": tool, "arguments": arguments or {}}}, timeout=timeout)
        return _decode_tool_result(tool, result or {}, time.perf_counter() - started, raise_on_error)

    def tools(self) -> list[dict]:
        return self.account.call("cloud_game_tools", {"session_id": self.session_id}).data.get("tools", [])

    def wait_ready(self, timeout_s: float = 300.0) -> None:
        """Until the game in the session answers status."""
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            try:
                answer = self.call("status", raise_on_error=False, timeout=30)
                if answer.data.get("app_state"):
                    return
                last = answer.data
            except McpError as e:
                last = str(e)
            time.sleep(3)
        raise TimeoutError(f"session {self.session_id} never answered status: {last}")
