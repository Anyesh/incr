"""Incremental computation graph for the API monitoring dashboard.

Builds an incr graph that processes simulated HTTP requests through
filtered collections and derived query nodes, then exposes metrics
and execution traces for the frontend.
"""

import random
import time
from collections import deque

from incr import Runtime

WINDOW_SIZE = 200  # rolling window of last N requests for rate/latency metrics

# ── Endpoint definitions for realistic traffic simulation ────────────────────

ENDPOINTS = [
    ("GET", "/api/users", 0.02, 45),
    ("GET", "/api/products", 0.01, 30),
    ("POST", "/api/orders", 0.05, 120),
    ("GET", "/api/health", 0.001, 5),
    ("PUT", "/api/users/:id", 0.03, 80),
    ("DELETE", "/api/sessions", 0.02, 25),
]


class DashboardEngine:
    def __init__(self):
        self.rt = Runtime()
        self.next_id = 0

        # Rolling window for rate and latency metrics (last N requests).
        # These are input nodes updated manually because collections are sets
        # and can't compute sums or windowed aggregates.
        self._recent = deque(maxlen=WINDOW_SIZE)
        self._win_error_count = 0
        self._win_latency_sum = 0.0
        self._win_slow_count = 0

        self._win_count_node = self.rt.create_input(0)
        self.rt.set_label(self._win_count_node, "recent_count")

        self._win_errors_node = self.rt.create_input(0)
        self.rt.set_label(self._win_errors_node, "recent_errors")

        self._win_latsum_node = self.rt.create_input(0.0)
        self.rt.set_label(self._win_latsum_node, "recent_lat_sum")

        self._win_slow_node = self.rt.create_input(0)
        self.rt.set_label(self._win_slow_node, "recent_slow")

        self._throughput_reqs_node = self.rt.create_input(0)
        self.rt.set_label(self._throughput_reqs_node, "window_reqs")

        self._throughput_secs_node = self.rt.create_input(10.0)
        self.rt.set_label(self._throughput_secs_node, "window_secs")

        # Base collection (tracks all-time totals)
        self.all_requests = self.rt.create_collection()
        self.rt.set_label_by_id(self.all_requests.version_node_id, "all_requests")

        # All-time counts via collection pipeline
        self.total_count = self.all_requests.count()
        self.rt.set_label(self.total_count, "total_count")

        # Derived query nodes using windowed inputs
        wc = self._win_count_node
        we = self._win_errors_node
        wl = self._win_latsum_node
        ws = self._win_slow_node
        treqs = self._throughput_reqs_node
        tsecs = self._throughput_secs_node

        self.error_rate = self.rt.create_query(
            lambda rt: rt.get(we) / max(rt.get(wc), 1) * 100
        )
        self.rt.set_label(self.error_rate, "error_rate")

        self.avg_latency = self.rt.create_query(
            lambda rt: rt.get(wl) / max(rt.get(wc), 1)
        )
        self.rt.set_label(self.avg_latency, "avg_latency")

        self.error_count = self.rt.create_query(lambda rt: rt.get(we))
        self.rt.set_label(self.error_count, "error_count")

        self.slow_count = self.rt.create_query(lambda rt: rt.get(ws))
        self.rt.set_label(self.slow_count, "slow_count")

        self.throughput = self.rt.create_query(
            lambda rt: rt.get(treqs) / max(rt.get(tsecs), 0.1)
        )
        self.rt.set_label(self.throughput, "throughput")

        er = self.error_rate
        al = self.avg_latency
        self.health = self.rt.create_query(
            lambda rt: (
                "healthy"
                if rt.get(er) < 5 and rt.get(al) < 300
                else "degraded"
                if rt.get(er) < 15
                else "critical"
            )
        )
        self.rt.set_label(self.health, "health_status")

        # Force initial computation so the graph is wired up
        self.rt.get(self.health)

        # Track window for throughput
        self._throughput_start = time.time()
        self._throughput_count = 0

        # Plain list for from-scratch comparison
        self._all_events = []

    def _update_window(self, new_event):
        # If the window is full, the oldest event is about to be evicted
        evicted = None
        if len(self._recent) == self._recent.maxlen:
            evicted = self._recent[0]

        self._recent.append(new_event)

        # Adjust running window aggregates
        status, latency = new_event[4], new_event[5]
        self._win_latency_sum += latency
        if status >= 400:
            self._win_error_count += 1
        if latency > 500:
            self._win_slow_count += 1

        if evicted:
            old_status, old_latency = evicted[4], evicted[5]
            self._win_latency_sum -= old_latency
            if old_status >= 400:
                self._win_error_count -= 1
            if old_latency > 500:
                self._win_slow_count -= 1

        # Push updated window values into incr input nodes
        win_count = len(self._recent)
        self.rt.set(self._win_count_node, win_count)
        self.rt.set(self._win_errors_node, self._win_error_count)
        self.rt.set(self._win_latsum_node, self._win_latency_sum)
        self.rt.set(self._win_slow_node, self._win_slow_count)

    def add_request(self, method: str, endpoint: str, status: int, latency: int):
        rid = self.next_id
        self.next_id += 1
        ts = int(time.time() * 1000)
        event = (rid, ts, method, endpoint, status, latency)
        self.all_requests.insert(event)

        self._update_window(event)

        # Update throughput window
        self._throughput_count += 1
        elapsed = time.time() - self._throughput_start
        if elapsed > 0.1:
            self.rt.set(self._throughput_reqs_node, self._throughput_count)
            self.rt.set(self._throughput_secs_node, elapsed)

        self._all_events.append(event)
        return event

    def recompute_from_scratch(self):
        # The naive approach without incr: you have all events in a list and
        # need to derive metrics. Without incremental tracking you'd scan the
        # full list each time (to find totals, filter errors, compute rates).
        events = self._all_events
        total = len(events)
        errors = sum(1 for e in events if e[4] >= 400)
        slow = sum(1 for e in events if e[5] > 500)
        latency_sum = sum(e[5] for e in events)
        win_total = total

        error_rate = (errors / max(win_total, 1)) * 100
        avg_latency = latency_sum / max(win_total, 1)
        elapsed = time.time() - self._throughput_start
        throughput = self._throughput_count / max(elapsed, 0.1)

        if error_rate < 5 and avg_latency < 300:
            health = "healthy"
        elif error_rate < 15:
            health = "degraded"
        else:
            health = "critical"

        return {
            "total_requests": total,
            "error_count": errors,
            "slow_count": slow,
            "error_rate": round(error_rate, 2),
            "avg_latency": round(avg_latency, 1),
            "throughput": round(throughput, 1),
            "health": health,
        }

    def read_all_traced(self):
        # get_traced on health (deepest node) captures the full propagation,
        # then plain get() for individual metric values (already clean).
        val_health, trace = self.rt.get_traced(self.health)

        metrics = {
            "total_requests": self.rt.get(self.total_count),
            "error_count": self.rt.get(self.error_count),
            "slow_count": self.rt.get(self.slow_count),
            "error_rate": round(self.rt.get(self.error_rate), 2),
            "avg_latency": round(self.rt.get(self.avg_latency), 1),
            "throughput": round(self.rt.get(self.throughput), 1),
            "health": val_health,
        }

        return metrics, trace

    def graph_snapshot(self):
        return self.rt.graph_snapshot()

    def node_count(self):
        return self.rt.node_count()


class TrafficSimulator:
    def __init__(self, engine: DashboardEngine):
        self.engine = engine
        self.mode = "normal"
        self.mode_remaining = 0
        self.recent_events = []

    def generate_batch(self, count: int = 2):
        events = []
        for _ in range(count):
            method, path, err_prob, base_lat = random.choice(ENDPOINTS)

            if self.mode == "error_burst":
                err_prob = 0.6
            elif self.mode == "latency_spike":
                base_lat *= 8
            elif self.mode == "traffic_surge":
                pass  # handled by increasing count externally

            if random.random() < err_prob:
                status = random.choice([400, 404, 500, 502, 503])
            else:
                status = 200

            latency = max(1, int(random.gauss(base_lat, base_lat * 0.3)))
            event = self.engine.add_request(method, path, status, latency)
            events.append(event)

        # Keep last 50 events for the UI feed
        self.recent_events.extend(events)
        self.recent_events = self.recent_events[-50:]

        if self.mode_remaining > 0:
            self.mode_remaining -= 1
            if self.mode_remaining == 0:
                self.mode = "normal"

        return events

    def trigger_scenario(self, scenario: str):
        self.mode = scenario
        self.mode_remaining = 15
