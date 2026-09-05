#!/usr/bin/env python3
"""Exhaustively verify benchmarks/minimum_facility.json."""
import itertools
import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "minimum_facility.json"

def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))

def solve(task):
    locations=task["locations"]; target=task["coverage_target_percent"]
    total=sum(x["population"] for x in locations); required=(target*total+99)//100
    for count in range(len(locations)+1):
        solutions=[]
        for centers in itertools.combinations(range(len(locations)),count):
            covered=tuple(i for i,x in enumerate(locations) if any(distance_km(locations[c],x)<=task["coverage_radius_km"]+1e-9 for c in centers))
            people=sum(locations[i]["population"] for i in covered)
            if people>=required: solutions.append((centers,covered,people))
        if solutions:
            return {"minimum_centers":count,"total_population":total,"minimum_population_required":required,"optimal_solutions":[{"center_locations":[locations[i]["name"] for i in centers],"covered_locations":[locations[i]["name"] for i in covered],"people_covered":people} for centers,covered,people in solutions]}
    raise RuntimeError("No feasible solution")

def main():
    tasks=json.loads(PATH.read_text()); passed=0
    if len(tasks)!=1000: raise ValueError("Expected exactly 1,000 tasks")
    for number,task in enumerate(tasks,1):
        computed=solve(task); ok=computed==task["answer"]; passed+=ok
        print(f"Task {number}: {'PASS' if ok else 'FAIL'} | minimum {computed['minimum_centers']} | {len(computed['optimal_solutions'])} optimal set(s)")
    print(f"\n{passed}/{len(tasks)} stored answers passed verification.")
    return 0 if passed==len(tasks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
