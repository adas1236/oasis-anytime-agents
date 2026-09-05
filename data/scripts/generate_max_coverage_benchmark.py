#!/usr/bin/env python3
"""Generate 1,000 deterministic maximum-coverage benchmark tasks."""
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSP_DATA = ROOT / "scripts" / "tsp_road_data.json"
OUTPUT = ROOT / "benchmarks" / "max_coverage.json"
RADII = {
    "United States": [10, 25, 50, 80],
    "England, United Kingdom": [10, 30, 60, 100],
    "Delhi, India": [3, 5, 8, 12],
    "Tokyo, Japan": [2, 4, 6, 10],
    "Cape Town, South Africa": [5, 10, 15, 25],
}

def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))

def planning_population(region, name):
    digest = hashlib.sha256(f"{region}|{name}|max-coverage-v1".encode()).digest()
    return 1000 + int.from_bytes(digest[:4], "big") % 24001

def solve(locations, centers_to_place, radius_km):
    best_covered = -1; solutions = []
    for centers in itertools.combinations(range(len(locations)), centers_to_place):
        covered = tuple(i for i, demand in enumerate(locations) if any(haversine_km(locations[c], demand) <= radius_km + 1e-9 for c in centers))
        total = sum(locations[i]["population"] for i in covered)
        if total > best_covered:
            best_covered, solutions = total, [(centers, covered)]
        elif total == best_covered:
            solutions.append((centers, covered))
    return best_covered, solutions

def main():
    road = json.loads(TSP_DATA.read_text())
    rng = random.Random(20260904)
    output = []
    for region, region_data in road["regions"].items():
        pool = [{"name": p["name"], "latitude": p["lat"], "longitude": p["lon"], "population": planning_population(region, p["name"])} for p in region_data["places"]]
        subsets = [s for size in range(6, 11) for s in itertools.combinations(range(10), size)]
        rng.shuffle(subsets)
        for local_number, subset in enumerate(subsets[:200]):
            locations = [pool[i] for i in subset]
            centers_to_place = 2 + local_number % min(3, len(locations)-1)
            radius_km = RADII[region][(local_number // 3) % len(RADII[region])]
            total, solutions = solve(locations, centers_to_place, radius_km)
            candidate_names = ", ".join(x["name"] for x in locations)
            prompt = (f"A vaccination program may place {centers_to_place} centers at the following candidate locations in {region}: {candidate_names}. "
                      f"Each center covers every listed population point within {radius_km} km geodesic distance. Using the coordinates and synthetic planning-population counts provided, "
                      "where should the centers be placed to maximize the number of people covered? Count each population point at most once.")
            output.append({
                "prompt": prompt,
                "centers_to_place": centers_to_place,
                "coverage_radius_km": radius_km,
                "distance_metric": "geodesic (haversine)",
                "population_data_type": "synthetic planning counts",
                "locations": locations,
                "answer": {
                    "people_covered": total,
                    "optimal_solutions": [
                        {"center_locations": [locations[i]["name"] for i in centers], "covered_locations": [locations[i]["name"] for i in covered]}
                        for centers, covered in solutions
                    ],
                },
                "geographic_region": region,
            })
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(output)} maximum-coverage benchmarks")

if __name__ == "__main__":
    main()
