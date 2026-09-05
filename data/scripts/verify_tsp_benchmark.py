#!/usr/bin/env python3
"""Independently recompute all exact solutions in benchmarks/tsp.json.

The companion tsp_road_data.json contains precise Nominatim-resolved places,
OSM feature IDs, directed OSRM driving-distance matrices, and instance mappings.
All routing data was retrieved 2026-09-03 and is attributed to OpenStreetMap
contributors under ODbL 1.0. No network access is needed for verification.
"""
import itertools, json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def exact_tsp(matrix,nodes):
    start=nodes[0]; best=float("inf"); best_order=None
    for middle in itertools.permutations(nodes[1:]):
        order=(start,*middle,start)
        total=sum(matrix[a][b] for a,b in zip(order,order[1:]))
        if total<best: best,best_order=total,order
    return best,best_order

def rounded_km(meters):
    return int((Decimal(str(meters))/Decimal("1000")).quantize(Decimal("1"),rounding=ROUND_HALF_UP))

def main():
    benchmarks=json.loads((ROOT/"benchmarks"/"tsp.json").read_text())
    road=json.loads((ROOT/"scripts"/"tsp_road_data.json").read_text())
    instances=road["instances"]
    if len(benchmarks)!=1000 or len(instances)!=1000: raise ValueError("Expected exactly 1,000 benchmarks and instances")
    passed=0
    for number,(benchmark,instance) in enumerate(zip(benchmarks,instances),1):
        region=instance["region"]; region_data=road["regions"][region]
        matrix=region_data["distances_meters"]; nodes=instance["nodes"]
        if len(set(nodes))!=len(nodes) or not 4<=len(nodes)<=7: raise ValueError(f"Invalid nodes in task {number}")
        expected_locations=[{"name":region_data["places"][i]["name"],"latitude":region_data["places"][i]["lat"],"longitude":region_data["places"][i]["lon"]} for i in nodes]
        if benchmark.get("locations")!=expected_locations: raise ValueError(f"Location mapping mismatch in task {number}")
        distance,order=exact_tsp(matrix,nodes)
        ok=isinstance(benchmark["answer"], int) and benchmark["answer"]==rounded_km(distance)
        passed+=ok
        names=[region_data["places"][i]["name"] for i in order]
        print(f"Task {number}: {'PASS' if ok else 'FAIL'} | {distance:.1f} m | {' -> '.join(names)}")
    print(f"\n{passed}/{len(benchmarks)} stored answers passed verification.")
    return 0 if passed==len(benchmarks) else 1

if __name__=="__main__": raise SystemExit(main())
