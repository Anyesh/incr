"""FastAPI server that runs the incr dashboard demo.

Serves the frontend, manages a background simulation loop, and pushes
incremental computation traces to connected WebSocket clients.
"""

import asyncio
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine import DashboardEngine, TrafficSimulator

app = FastAPI()
engine = DashboardEngine()
simulator = TrafficSimulator(engine)
connected_clients: list[WebSocket] = []

# Pre-seed with 5,000 events so the incremental vs from-scratch comparison
# is meaningful from the moment a visitor opens the page. All events go
# through the real incr pipeline; nothing is faked.
for _ in range(2500):
    simulator.generate_batch(2)
engine.read_all_traced()  # stabilize the graph


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)

    # Send the graph structure once on connect
    try:
        await ws.send_json(
            {"type": "graph_structure", "nodes": engine.graph_snapshot()}
        )
    except Exception:
        connected_clients.remove(ws)
        return

    # Listen for control messages from the client
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "trigger_scenario":
                simulator.trigger_scenario(data["scenario"])
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def simulation_loop():
    tick = 0
    while True:
        batch_size = 5 if simulator.mode == "traffic_surge" else 2
        events = simulator.generate_batch(batch_size)

        # Read metrics with tracing (includes trace overhead, used for graph viz)
        metrics, trace = engine.read_all_traced()

        # Measure the per-update cost of each approach by doing one more
        # insert and timing how long it takes to get fresh metrics.
        engine.add_request("GET", "/api/health", 200, 5)

        start_ns = time.perf_counter_ns()
        engine.rt.get(engine.health)
        engine.rt.get(engine.total_count)
        engine.rt.get(engine.error_count)
        engine.rt.get(engine.error_rate)
        engine.rt.get(engine.avg_latency)
        incremental_us = (time.perf_counter_ns() - start_ns) / 1000

        start_ns = time.perf_counter_ns()
        engine.recompute_from_scratch()
        scratch_us = (time.perf_counter_ns() - start_ns) / 1000

        # Build the trace summary from the traced get() call
        recomputed_ids = set()
        cutoff_ids = set()
        visited_ids = set()
        for nt in trace.get("node_traces", []):
            visited_ids.add(nt["id"])
            if nt["action"] == "recomputed_changed":
                recomputed_ids.add(nt["id"])
            elif nt["action"] == "recomputed_cutoff":
                cutoff_ids.add(nt["id"])
            elif nt["action"] == "verified_clean":
                visited_ids.add(nt["id"])

        # Format recent events for the feed (last 10)
        recent = []
        for ev in simulator.recent_events[-10:]:
            recent.append(
                {
                    "id": ev[0],
                    "method": ev[2],
                    "endpoint": ev[3],
                    "status": ev[4],
                    "latency": ev[5],
                }
            )

        msg = {
            "type": "update",
            "tick": tick,
            "timestamp": int(time.time() * 1000),
            "metrics": metrics,
            "trace": {
                "total_nodes": engine.node_count(),
                "nodes_recomputed": len(recomputed_ids),
                "nodes_cutoff": len(cutoff_ids),
                "nodes_visited": len(visited_ids),
                "incremental_us": round(incremental_us, 1),
                "scratch_us": round(scratch_us, 1),
                "speedup": round(scratch_us / max(incremental_us, 0.1), 1),
                "recomputed_ids": list(recomputed_ids),
                "cutoff_ids": list(cutoff_ids),
                "visited_ids": list(visited_ids),
            },
            "scenario": simulator.mode,
            "recent_events": recent,
        }

        # Broadcast to all connected clients
        disconnected = []
        for ws in connected_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            connected_clients.remove(ws)

        tick += 1
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())
