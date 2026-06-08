"""Plumbing smoke for the gate→web integration (no torch/LLM): mock the agent, drive the real
AgentHTTPServer, and assert the SSE pub/sub + /api/gate POST behave (status broadcast, wake →
agent.load_mat + alert + ecg_ready)."""
import http.client
import json
import os
import tempfile
import threading
import time

import web_server


class FakeAgent:
    def __init__(self):
        self.mat_path = "init.mat"
        self.loaded = []

    def load_mat(self, p):
        self.loaded.append(p)
        self.mat_path = p

    def summary(self):
        return {"classification": []}

    def waveform(self):
        return {"leads": []}


def _post(port, obj):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    c.request("POST", "/api/gate", body=json.dumps(obj),
              headers={"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())


def main():
    agent = FakeAgent()
    srv = web_server.AgentHTTPServer(("127.0.0.1", 0), web_server.Handler, agent)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # subscribe an SSE client and collect events in a background reader
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/api/gate/stream")
    resp = conn.getresponse()
    events = []

    def reader():
        for raw in iter(resp.fp.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except Exception:
                    pass

    threading.Thread(target=reader, daemon=True).start()
    time.sleep(0.2)  # let the initial state event arrive

    # 1) status POST -> broadcast + gate_state updated
    assert _post(port, {"type": "status", "state": "armed", "score": 0.21, "thr": 0.28})["ok"]
    # 2) wake POST with a real temp .mat -> agent.load_mat + wake + ecg_ready
    tmp = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
    tmp.write(b"x")
    tmp.close()
    assert _post(port, {"type": "wake", "mat": tmp.name, "score": 0.30, "t": 150})["ok"]
    time.sleep(0.3)

    types = [e.get("type") for e in events]
    print("events:", types)
    assert types[0] == "status" and events[0].get("state") == "idle", f"first = current state: {events[0]}"
    assert any(e.get("type") == "status" and e.get("score") == 0.21 for e in events), "status broadcast"
    assert "wake" in types, "wake broadcast"
    assert "ecg_ready" in types, "ecg_ready after load_mat"
    assert agent.loaded == [tmp.name], f"agent.load_mat called with wake .mat: {agent.loaded}"
    # ecg_ready must come AFTER wake (alert first, then refresh)
    assert types.index("wake") < types.index("ecg_ready"), "alert before ecg_ready"
    assert srv.gate_state.get("state") == "alarm", "gate_state reflects last alarm"

    # 3) wake with a missing .mat -> gate_error, no extra load
    assert _post(port, {"type": "wake", "mat": "/no/such.mat", "score": 0.9, "t": 99})["ok"]
    time.sleep(0.2)
    assert any(e.get("type") == "gate_error" for e in events), "missing .mat -> gate_error"
    assert agent.loaded == [tmp.name], "no load on missing .mat"

    os.unlink(tmp.name)
    srv.shutdown()
    print("\nGATE-WEB PLUMBING SMOKE PASSED")


if __name__ == "__main__":
    main()
