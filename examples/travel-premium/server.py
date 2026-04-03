"""FastAPI server for the travel premium demo.

Serves the frontend, manages the engine, and pushes incremental vs
from-scratch results to connected WebSocket clients.
"""

import asyncio
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine import TravelPremiumEngine, Visit, _random_location, CLIENT_NAMES, ADDRESSES

app = FastAPI()
engine = TravelPremiumEngine()
engine.distance_service.latency_range = (30, 50)
engine.generate_initial_schedule(14)


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Send initial state
    try:
        result = engine.read_incremental()
        await ws.send_json({"type": "initial", **result})
    except Exception:
        return

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "add_visit":
                engine.add_visit()

            elif action == "remove_visit":
                engine.remove_visit()

            elif action == "reschedule_visit":
                engine.reschedule_visit()

            elif action == "midday_insertion":
                lat, lng = _random_location()
                engine.add_visit(
                    Visit(
                        id=engine.next_id,
                        client_name=random.choice(CLIENT_NAMES),
                        address=random.choice(ADDRESSES),
                        time=12.0,
                        duration_mins=30,
                        lat=lat,
                        lng=lng,
                    )
                )

            elif action == "morning_shuffle":
                morning = [v for v in engine.visits.values() if v.time < 12.0]
                for v in random.sample(morning, min(3, len(morning))):
                    engine.reschedule_visit(v.id)

            elif action == "cancel_afternoon":
                afternoon = [v for v in engine.visits.values() if v.time >= 13.0]
                for v in random.sample(afternoon, min(3, len(afternoon))):
                    engine.remove_visit(v.id)

            elif action == "reset":
                engine.__init__()
                engine.distance_service.latency_range = (30, 50)
                engine.generate_initial_schedule(14)
                result = engine.read_incremental()
                await ws.send_json({"type": "initial", **result})
                continue

            else:
                continue

            # First message: incremental result (fast)
            result = engine.read_incremental()
            await ws.send_json({"type": "incremental", **result})

            # Second message: from-scratch comparison (slow, runs in thread)
            scratch = await asyncio.to_thread(engine.compute_from_scratch)
            await ws.send_json({"type": "scratch", **scratch})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
