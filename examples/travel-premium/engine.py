"""Incremental travel premium computation engine.

Builds an incr pipeline that processes an employee's daily visit schedule,
computes travel distances between consecutive visits, and calculates
travel premiums incrementally.
"""

import math
import random
import time
from dataclasses import dataclass, field

from incr import Runtime


# ── Data Model ───────────────────────────────────────────────────────────────


@dataclass
class Visit:
    id: int
    client_name: str
    address: str
    time: float  # hour of day, e.g. 9.5 = 9:30am
    duration_mins: int
    lat: float
    lng: float

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Visit) and self.id == other.id


CLIENT_NAMES = [
    "Margaret Thompson",
    "James Wilson",
    "Dorothy Chen",
    "Robert Garcia",
    "Helen Patel",
    "William Brown",
    "Patricia Kim",
    "Charles Davis",
    "Barbara Singh",
    "Thomas Moore",
    "Susan Clark",
    "Richard Lee",
    "Nancy White",
    "David Hall",
    "Linda Young",
    "Michael Taylor",
]

ADDRESSES = [
    "12 Oak Street",
    "45 Maple Avenue",
    "78 Cedar Lane",
    "23 Elm Drive",
    "56 Pine Road",
    "89 Birch Way",
    "34 Willow Court",
    "67 Spruce Place",
    "90 Ash Boulevard",
    "11 Poplar Circle",
    "44 Cherry Street",
    "77 Walnut Ave",
    "22 Hickory Lane",
    "55 Chestnut Drive",
    "88 Sycamore Road",
    "33 Linden Way",
]

# City center coordinates (roughly Auckland, NZ)
CENTER_LAT = -36.85
CENTER_LNG = 174.76
CITY_RADIUS_KM = 15


def _random_location():
    angle = random.uniform(0, 2 * math.pi)
    dist_km = random.uniform(0, CITY_RADIUS_KM)
    dlat = (dist_km * math.cos(angle)) / 111.0
    dlng = (dist_km * math.sin(angle)) / (111.0 * math.cos(math.radians(CENTER_LAT)))
    return round(CENTER_LAT + dlat, 6), round(CENTER_LNG + dlng, 6)


def generate_schedule(num_visits: int = 14) -> list[Visit]:
    visits = []
    slot_duration = 10.0 / num_visits
    for i in range(num_visits):
        base_time = 7.0 + i * slot_duration
        time_val = round(base_time + random.uniform(0, slot_duration * 0.6), 2)
        time_val = min(time_val, 16.75)

        lat, lng = _random_location()
        visit = Visit(
            id=i,
            client_name=random.choice(CLIENT_NAMES),
            address=random.choice(ADDRESSES),
            time=time_val,
            duration_mins=random.choice([15, 30, 30, 45, 60]),
            lat=lat,
            lng=lng,
        )
        visits.append(visit)
    return visits


# ── Distance Service ─────────────────────────────────────────────────────────


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _round_coord(v: float) -> float:
    # Round to ~100m precision for cache keys.
    return round(v, 3)


class DistanceService:
    def __init__(self, latency_ms: tuple[int, int] = (30, 50)):
        self.latency_range = latency_ms
        self.cache: dict[tuple, float] = {}
        self.stats = {"hits": 0, "misses": 0, "total_ms": 0.0}

    def reset_stats(self):
        self.stats = {"hits": 0, "misses": 0, "total_ms": 0.0}

    def clear_cache(self):
        self.cache.clear()

    def get_distance(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> tuple[float, bool]:
        # Returns (distance_km, cache_hit).
        key = (
            _round_coord(lat1),
            _round_coord(lng1),
            _round_coord(lat2),
            _round_coord(lng2),
        )

        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key], True

        delay_ms = random.randint(*self.latency_range)
        time.sleep(delay_ms / 1000.0)

        dist = haversine_km(lat1, lng1, lat2, lng2)
        self.cache[key] = dist
        self.stats["misses"] += 1
        self.stats["total_ms"] += delay_ms
        return dist, False


# ── Premium Rules ────────────────────────────────────────────────────────────

DAILY_CAP = 50.0


def compute_premium(distance_km: float) -> float:
    # Tiered: 0-5km free, 5-20km at $0.45/km, 20+km at $0.65/km.
    if distance_km <= 5.0:
        return 0.0
    elif distance_km <= 20.0:
        return (distance_km - 5.0) * 0.45
    else:
        return (15.0 * 0.45) + (distance_km - 20.0) * 0.65


@dataclass
class Segment:
    from_id: int
    to_id: int
    distance_km: float
    premium: float
    cache_hit: bool

    def __hash__(self):
        return hash((self.from_id, self.to_id))

    def __eq__(self, other):
        return (
            isinstance(other, Segment)
            and self.from_id == other.from_id
            and self.to_id == other.to_id
        )


# ── incr Pipeline Engine ─────────────────────────────────────────────────────


class TravelPremiumEngine:
    def __init__(self):
        self.rt = Runtime()
        self.distance_service = DistanceService()
        self.visits: dict[int, Visit] = {}
        self.next_id = 0

        # Build the incr pipeline
        self.visit_collection = self.rt.create_collection()

        self.sorted_visits = self.visit_collection.sort_by_key(lambda v: v.time)
        self.pairs = self.sorted_visits.pairwise()

        svc = self.distance_service

        def _pair_to_segment(pair):
            a, b = pair[0], pair[1]
            dist, hit = svc.get_distance(a.lat, a.lng, b.lat, b.lng)
            return Segment(
                from_id=a.id,
                to_id=b.id,
                distance_km=dist,
                premium=compute_premium(dist),
                cache_hit=hit,
            )

        self.segments = self.pairs.map(_pair_to_segment)

        self.total_premium = self.segments.reduce(
            lambda segs: min(sum(s.premium for s in segs), DAILY_CAP)
        )

    def generate_initial_schedule(self, num_visits: int = 14):
        schedule = generate_schedule(num_visits)
        for v in schedule:
            self.visit_collection.insert(v)
            self.visits[v.id] = v
        self.next_id = num_visits
        self.rt.get(self.total_premium)

    def add_visit(self, visit: Visit = None) -> Visit:
        if visit is None:
            lat, lng = _random_location()
            visit = Visit(
                id=self.next_id,
                client_name=random.choice(CLIENT_NAMES),
                address=random.choice(ADDRESSES),
                time=round(random.uniform(7.5, 16.5), 2),
                duration_mins=random.choice([15, 30, 30, 45, 60]),
                lat=lat,
                lng=lng,
            )
        self.next_id = max(self.next_id, visit.id + 1)
        self.visit_collection.insert(visit)
        self.visits[visit.id] = visit
        return visit

    def remove_visit(self, visit_id: int = None) -> Visit | None:
        if not self.visits:
            return None
        if visit_id is None:
            visit_id = random.choice(list(self.visits.keys()))
        visit = self.visits.pop(visit_id, None)
        if visit:
            self.visit_collection.delete(visit)
        return visit

    def reschedule_visit(self, visit_id: int = None) -> tuple[Visit, Visit] | None:
        if not self.visits:
            return None
        if visit_id is None:
            visit_id = random.choice(list(self.visits.keys()))
        old = self.visits.get(visit_id)
        if not old:
            return None
        self.visit_collection.delete(old)
        new = Visit(
            id=old.id,
            client_name=old.client_name,
            address=old.address,
            time=round(random.uniform(7.5, 16.5), 2),
            duration_mins=old.duration_mins,
            lat=old.lat,
            lng=old.lng,
        )
        self.visit_collection.insert(new)
        self.visits[new.id] = new
        return old, new

    def read_incremental(self) -> dict:
        self.distance_service.reset_stats()

        start = time.perf_counter()
        total = self.rt.get(self.total_premium)
        elapsed_ms = (time.perf_counter() - start) * 1000

        stats = self.distance_service.stats.copy()

        return {
            "total_premium": round(total, 2),
            "visits": [
                {
                    "id": v.id,
                    "client_name": v.client_name,
                    "address": v.address,
                    "time": v.time,
                    "duration_mins": v.duration_mins,
                    "lat": v.lat,
                    "lng": v.lng,
                }
                for v in sorted(self.visits.values(), key=lambda v: v.time)
            ],
            "distance_stats": stats,
            "incremental_ms": round(elapsed_ms, 1),
        }

    def compute_from_scratch(self) -> dict:
        self.distance_service.clear_cache()
        self.distance_service.reset_stats()

        start = time.perf_counter()

        sorted_v = sorted(self.visits.values(), key=lambda v: v.time)
        total = 0.0
        for i in range(len(sorted_v) - 1):
            a, b = sorted_v[i], sorted_v[i + 1]
            dist, _ = self.distance_service.get_distance(a.lat, a.lng, b.lat, b.lng)
            total += compute_premium(dist)
        total = min(total, DAILY_CAP)

        elapsed_ms = (time.perf_counter() - start) * 1000
        stats = self.distance_service.stats.copy()

        return {
            "total_premium": round(total, 2),
            "distance_stats": stats,
            "scratch_ms": round(elapsed_ms, 1),
        }
