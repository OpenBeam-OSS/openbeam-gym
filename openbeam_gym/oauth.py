"""OAuth for the OpenBeam account MCP endpoint, as a native MCP client does it.

The flow any native MCP client uses with OpenBeam (the web console's own MCP
client does the same):

1. discovery: an unauthenticated POST /mcp answers 401 with a
   WWW-Authenticate resource_metadata URL; the protected-resource metadata
   names the authorization server, whose metadata names the endpoints;
2. dynamic client registration (RFC 7591) as a public client with a loopback
   redirect `http://127.0.0.1:<port>/callback`;
3. authorization code with PKCE S256, scope `mcp:tools`, `resource` = the MCP
   URL: the person approves on OpenBeam's own page in their browser
   (session-control permission included, so the client can start sessions);
4. tokens are cached owner-only in ~/.config/openbeam-gym/oauth-<host>.json
   and refreshed with the refresh token.

Headless use: set OPENBEAM_MCP_ACCESS_TOKEN (an access token minted by a
previous approval) or pass `token_file`. Tokens are never printed.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

SCOPE = "mcp:tools"


def _json(url: str, data: bytes | None = None, headers: dict | None = None, method: str | None = None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            return response.status, dict(response.headers), json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body) if body else {}
        except ValueError:
            parsed = {"raw": body[:300].decode("utf-8", "replace")}
        return e.code, dict(e.headers), parsed


class OAuthTokens:
    """An access-token provider for one MCP base URL."""

    def __init__(self, base_url: str, cache_dir: str | Path = "~/.config/openbeam-gym",
                 token_file: str | Path | None = None, open_browser=webbrowser.open, approve=None,
                 callback_port: int = 0, client_name: str = "openbeam-gym"):
        self.base_url = base_url.rstrip("/")
        self.mcp_url = f"{self.base_url}/mcp"
        host = urllib.parse.urlparse(self.base_url).netloc.replace(":", "_")
        self.cache = Path(cache_dir).expanduser() / f"oauth-{host}.json"
        self.token_file = Path(token_file).expanduser() if token_file else None
        self.open_browser = open_browser
        self.approve = approve          # optional callable(url) for automated approval (tests, harness)
        self.callback_port = callback_port
        self.client_name = client_name
        self._lock = threading.Lock()
        self._state: dict = {}

    # -- public ---------------------------------------------------------------
    def access_token(self) -> str:
        env = os.environ.get("OPENBEAM_MCP_ACCESS_TOKEN")
        if env:
            return env.strip()
        if self.token_file:
            return self.token_file.read_text().strip()
        with self._lock:
            if not self._state:
                self._load()
            if self._state.get("access_token") and self._state.get("expires_at", 0) > time.time() + 60:
                return self._state["access_token"]
            if self._state.get("refresh_token") and self._refresh():
                return self._state["access_token"]
            self._authorize()
            return self._state["access_token"]

    # -- cache ----------------------------------------------------------------
    def _load(self) -> None:
        try:
            self._state = json.loads(self.cache.read_text())
        except (OSError, ValueError):
            self._state = {}

    def _save(self) -> None:
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._state, f)
        os.replace(tmp, self.cache)

    # -- discovery --------------------------------------------------------------
    def _metadata(self) -> tuple[dict, str]:
        status, headers, _ = _json(self.mcp_url, json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                                             "params": {}}).encode(),
                                   {"Content-Type": "application/json", "Accept": "application/json"}, "POST")
        challenge = headers.get("WWW-Authenticate") or headers.get("www-authenticate") or ""
        match = re.search(r'resource_metadata="([^"]+)"', challenge)
        parsed = urllib.parse.urlparse(self.mcp_url)
        candidates = ([match.group(1)] if match else []) + [
            f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{parsed.path}",
            f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource"]
        resource_meta = None
        for url in candidates:
            s, _, body = _json(url)
            if s == 200 and body.get("authorization_servers"):
                resource_meta = body
                break
        if resource_meta is None:
            raise RuntimeError("the MCP server published no protected-resource metadata")
        issuer = urllib.parse.urlparse(resource_meta["authorization_servers"][0])
        as_meta = None
        for url in (f"{issuer.scheme}://{issuer.netloc}/.well-known/oauth-authorization-server{issuer.path.rstrip('/')}",
                    f"{issuer.scheme}://{issuer.netloc}/.well-known/oauth-authorization-server"):
            s, _, body = _json(url)
            if s == 200 and body.get("authorization_endpoint") and body.get("token_endpoint"):
                as_meta = body
                break
        if as_meta is None:
            raise RuntimeError("no authorization-server metadata")
        return as_meta, str(resource_meta.get("resource", self.mcp_url))

    # -- flows ------------------------------------------------------------------
    def _refresh(self) -> bool:
        try:
            as_meta, resource = self._metadata()
        except RuntimeError:
            return False
        status, _, body = _json(as_meta["token_endpoint"], urllib.parse.urlencode({
            "grant_type": "refresh_token", "refresh_token": self._state["refresh_token"],
            "client_id": self._state.get("client_id", ""), "resource": resource}).encode(),
            {"Content-Type": "application/x-www-form-urlencoded"}, "POST")
        if status != 200 or not body.get("access_token"):
            return False
        self._accept(body)
        return True

    def _accept(self, body: dict) -> None:
        self._state["access_token"] = body["access_token"]
        if body.get("refresh_token"):
            self._state["refresh_token"] = body["refresh_token"]
        self._state["expires_at"] = time.time() + float(body.get("expires_in", 3600))
        self._save()

    def _authorize(self) -> None:
        as_meta, resource = self._metadata()
        received: dict = {}
        done = threading.Event()

        class Callback(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                url = urllib.parse.urlparse(self.path)
                if url.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                received.update({k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()})
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>openbeam-gym is connected. You can close this tab.</h2>")
                done.set()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", self.callback_port), Callback)
        redirect = f"http://127.0.0.1:{server.server_address[1]}/callback"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            if not as_meta.get("registration_endpoint"):
                raise RuntimeError("the authorization server has no dynamic client registration endpoint")
            status, _, reg = _json(as_meta["registration_endpoint"], json.dumps({
                "client_name": self.client_name, "redirect_uris": [redirect],
                "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
                "token_endpoint_auth_method": "none", "scope": SCOPE}).encode(),
                {"Content-Type": "application/json"}, "POST")
            if status not in (200, 201) or "client_id" not in reg:
                raise RuntimeError(f"client registration failed (HTTP {status})")
            client_id = reg["client_id"]
            verifier = secrets.token_urlsafe(48)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            state = secrets.token_hex(16)
            query = urllib.parse.urlencode({"response_type": "code", "client_id": client_id, "redirect_uri": redirect,
                                            "code_challenge": challenge, "code_challenge_method": "S256",
                                            "scope": SCOPE, "state": state, "resource": resource})
            url = f"{as_meta['authorization_endpoint']}?{query}"
            if self.approve is not None:
                self.approve(url)
            else:
                print("openbeam-gym: approve the connection in your browser (OpenBeam's page).", flush=True)
                self.open_browser(url)
            if not done.wait(300):
                raise TimeoutError("no authorization callback within 5 minutes")
            if received.get("state") != state:
                raise RuntimeError("OAuth state mismatch")
            if received.get("error"):
                raise RuntimeError(f"authorization refused: {received['error']}")
            status, _, tokens = _json(as_meta["token_endpoint"], urllib.parse.urlencode({
                "grant_type": "authorization_code", "code": received["code"], "redirect_uri": redirect,
                "client_id": client_id, "code_verifier": verifier, "resource": resource}).encode(),
                {"Content-Type": "application/x-www-form-urlencoded"}, "POST")
            if status != 200 or not tokens.get("access_token"):
                raise RuntimeError(f"token exchange failed (HTTP {status})")
            self._state = {"client_id": client_id}
            self._accept(tokens)
        finally:
            server.shutdown()
