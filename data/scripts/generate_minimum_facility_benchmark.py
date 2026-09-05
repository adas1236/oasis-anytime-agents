#!/usr/bin/env python3
"""Generate 1,000 deterministic minimum-facility/set-cover tasks."""
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "tsp_road_data.json"
OUTPUT = ROOT / "benchmarks" / "minimum_facility.json"
COVERAGE_PERCENT = 95
RADII = {
    "United States": [10, 25, 50, 80],
    "England, United Kingdom": [10, 30, 60, 100],
    "Delhi, India": [3, 5, 8, 12],
    "Tokyo, Japan": [2, 4, 6, 10],
    "Cape Town, South Africa": [5, 10, 15, 25],
}

def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))

def population(region, name):
    digest = hashlib.sha256(f"{region}|{name}|max-coverage-v1".encode()).digest()
    return 1000 + int.from_bytes(digest[:4], "big") % 24001

def solve(locations, radius_km):
    total_population = sum(x["population"] for x in locations)
    required_population = (COVERAGE_PERCENT * total_population + 99) // 100
    for count in range(len(locations) + 1):
        solutions = []
        for centers in itertools.combinations(range(len(locations)), count):
            covered = tuple(i for i, demand in enumerate(locations) if any(distance_km(locations[c], demand) <= radius_km + 1e-9 for c in centers))
            covered_population = sum(locations[i]["population"] for i in covered)
            if covered_population >= required_population:
                solutions.append((centers, covered, covered_population))
        if solutions:
            return required_population, solutions
    raise RuntimeError("No feasible solution")

def main():
    road = json.loads(SOURCE.read_text())
    rng = random.Random(20260905)
    tasks = []
    for region, region_data in road["regions"].items():
        pool = [{"name": p["name"], "latitude": p["lat"], "longitude": p["lon"], "population": population(region, p["name"])} for p in region_data["places"]]
        subsets = [s for size in range(6, 11) for s in itertools.combinations(range(10), size)]
        rng.shuffle(subsets)
        for local_number, subset in enumerate(subsets[:200]):
            locations = [pool[i] for i in subset]
            radius_km = RADII[region][local_number % len(RADII[region])]
            required, solutions = solve(locations, radius_km)
            names = ", ".join(x["name"] for x in locations)
            prompt = (f"Vaccination centers may be opened only at these candidate locations in {region}: {names}. Each center serves every listed population point within {radius_km} km geodesic distance. "
                      f"Using the coordinates and synthetic planning-population counts provided, what is the minimum number of centers needed so that at least {COVERAGE_PERCENT}% of residents are covered? Count each population point at most once.")
            tasks.append({
                "prompt": prompt,
                "coverage_target_percent": COVERAGE_PERCENT,
                "coverage_radius_km": radius_km,
                "distance_metric": "geodesic (haversine)",
                "population_data_type": "synthetic planning counts",
                "locations": locations,
                "answer": {
                    "minimum_centers": len(solutions[0][0]),
                    "total_population": sum(x["population"] for x in locations),
                    "minimum_population_required": required,
                    "optimal_solutions": [{
                        "center_locations": [locations[i]["name"] for i in centers],
                        "covered_locations": [locations[i]["name"] for i in covered],
                        "people_covered": covered_population,
                    } for centers, covered, covered_population in solutions],
                },
                "geographic_region": region,
            })
    OUTPUT.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(tasks)} minimum-facility benchmarks")

if __name__ == "__main__":
    main()
