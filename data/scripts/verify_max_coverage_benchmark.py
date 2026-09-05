#!/usr/bin/env python3
"""Exactly verify every task in benchmarks/max_coverage.json."""
import itertools
import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "max_coverage.json"

def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))

def exact_solution(task):
    locations=task["locations"]; radius=task["coverage_radius_km"]
    best=-1; solutions=[]
    for centers in itertools.combinations(range(len(locations)), task["centers_to_place"]):
        covered=tuple(i for i,x in enumerate(locations) if any(distance_km(locations[c],x)<=radius+1e-9 for c in centers))
        total=sum(locations[i]["population"] for i in covered)
        if total>best: best,solutions=total,[(centers,covered)]
        elif total==best: solutions.append((centers,covered))
    return {"people_covered":best,"optimal_solutions":[{"center_locations":[locations[i]["name"] for i in centers],"covered_locations":[locations[i]["name"] for i in covered]} for centers,covered in solutions]}

def main():
    tasks=json.loads(PATH.read_text()); passed=0
    if len(tasks)!=1000: raise ValueError("Expected exactly 1,000 tasks")
    for number,task in enumerate(tasks,1):
        computed=exact_solution(task); ok=computed==task["answer"]; passed+=ok
        print(f"Task {number}: {'PASS' if ok else 'FAIL'} | {computed['people_covered']} people | {len(computed['optimal_solutions'])} optimal set(s)")
    print(f"\n{passed}/{len(tasks)} stored answers passed verification.")
    return 0 if passed==len(tasks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
