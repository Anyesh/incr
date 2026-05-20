"""End-to-end verification of incr-spreadsheet against the v0.2 wrappers.

1. Connect to ws://localhost:3001/ws
2. Receive the full_state on connect; assert seed cells have expected
   computed values:
   - C2 = A2 * B2 = 29.99 * 5 = 149.95
   - C3 = A3 * B3 = 49.99 * 3 = 149.97
   - C4 = A4 * B4 =  9.99 * 12 = 119.88
   - C6 = SUM(C2:C4) = 419.80
   - C7 = C6 * 0.08 = 33.584
   - C8 = C6 + C7 = 453.384
3. Send set_cell A2 := 100; receive update; assert C2 -> 500,
   C6, C7, C8 update accordingly.
4. Send set_cell B2 := 10; receive update; assert C2 -> 1000.
5. Reset A2 -> 29.99; assert C2 back to 149.95.
"""

import asyncio
import json
import sys

import websockets


EXPECTED_SEED = {
    "C2": 149.95,
    "C3": 149.97,
    "C4": 119.88,
    "C6": 419.80,
    "C7": 33.584,
    "C8": 453.384,
}


def approx_eq(a, b, tol=1e-2):
    return abs(a - b) < tol


async def main():
    failures = []

    async with websockets.connect("ws://localhost:3001/ws") as ws:
        # Step 1: full_state on connect.
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        full = json.loads(raw)
        assert full["type"] == "full_state", full
        cells = {c["cell"]: c for c in full["cells"]}
        print(
            f"connected; node_count = {full['node_count']}, cell_count = {len(cells)}"
        )

        # Step 2: validate seed cells.
        for cell, want in EXPECTED_SEED.items():
            got = cells[cell]["value"]
            ok = approx_eq(got, want)
            status = "OK" if ok else "FAIL"
            print(f"  seed {cell}: got {got:.4f} want {want:.4f} [{status}]")
            if not ok:
                failures.append(f"seed {cell}: got {got}, want {want}")

        # Step 3: set A2 = 100.
        await ws.send(json.dumps({"cell": "A2", "content": "100"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        update = json.loads(raw)
        assert update["type"] == "update", update
        changed = {c["cell"]: c for c in update["changed"]}
        print(f"\nset A2 = 100; changed cells: {sorted(changed.keys())}")
        # After A2=100: C2 = 100 * 5 = 500
        # C6 = 500 + 149.97 + 119.88 = 769.85
        # C7 = 769.85 * 0.08 = 61.588
        # C8 = 769.85 + 61.588 = 831.438
        for cell, want in [
            ("C2", 500.0),
            ("C6", 769.85),
            ("C7", 61.588),
            ("C8", 831.438),
        ]:
            got = changed.get(cell, {}).get("value")
            ok = got is not None and approx_eq(got, want)
            status = "OK" if ok else "FAIL"
            print(f"  after A2=100 -> {cell}: got {got} want {want:.4f} [{status}]")
            if not ok:
                failures.append(f"A2=100 then {cell}: got {got}, want {want}")

        # Step 4: set B2 = 10.
        await ws.send(json.dumps({"cell": "B2", "content": "10"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        update = json.loads(raw)
        changed = {c["cell"]: c for c in update["changed"]}
        # C2 = 100 * 10 = 1000
        for cell, want in [("C2", 1000.0)]:
            got = changed.get(cell, {}).get("value")
            ok = got is not None and approx_eq(got, want)
            status = "OK" if ok else "FAIL"
            print(f"  after B2=10  -> {cell}: got {got} want {want:.4f} [{status}]")
            if not ok:
                failures.append(f"B2=10 then {cell}: got {got}, want {want}")

        # Step 5: reset A2 -> 29.99.
        await ws.send(json.dumps({"cell": "A2", "content": "29.99"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        update = json.loads(raw)
        changed = {c["cell"]: c for c in update["changed"]}
        # C2 = 29.99 * 10 = 299.9 (B2 is still 10 from step 4)
        got = changed.get("C2", {}).get("value")
        want = 299.9
        ok = got is not None and approx_eq(got, want)
        status = "OK" if ok else "FAIL"
        print(f"  reset A2=29.99 -> C2: got {got} want {want:.4f} [{status}]")
        if not ok:
            failures.append(f"reset A2 then C2: got {got}, want {want}")

    if failures:
        print(f"\nFAILED: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED")


asyncio.run(main())
