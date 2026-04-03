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
