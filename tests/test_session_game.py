"""SessionGame speaks MCP to /v1/sessions/<id>/mcp with its bearer token."""

import http.server
import json
import threading

from openbeam_gym import SessionGame


def test_session_game_posts_to_the_session_endpoint_with_its_token():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body.get("method")))
            if "id" not in body:
                self.send_response(202)
                self.end_headers()
                return
            result = {"content": [{"type": "text", "text": "{}"}], "structuredContent": {"ok": True}, "isError": False}
            data = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        game = SessionGame(f"http://127.0.0.1:{server.server_address[1]}", "trn_x", lambda: "secret-test-token")
        assert game.call("robot_state").data == {"ok": True}
    finally:
        server.shutdown()
    assert seen[-1] == ("/v1/sessions/trn_x/mcp", "Bearer secret-test-token", "tools/call")
