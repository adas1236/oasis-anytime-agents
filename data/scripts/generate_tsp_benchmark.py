#!/usr/bin/env python3
"""Generate 1,000 deterministic TSP benchmarks from live OSM/OSRM data."""
import itertools, json, random, time, urllib.parse, urllib.request
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOLS = {
"United States": [
("Johns Hopkins Hospital", "Johns Hopkins Hospital, Baltimore, Maryland, USA"),
("Anne Arundel Medical Center", "Anne Arundel Medical Center, Annapolis, Maryland, USA"),
("Frederick City Hall", "Frederick City Hall, Frederick, Maryland, USA"),
("Suburban Hospital", "Suburban Hospital, Bethesda, Maryland, USA"),
("University of Maryland Medical Center", "University of Maryland Medical Center, Baltimore, USA"),
("MedStar Washington Hospital Center", "MedStar Washington Hospital Center, Washington DC, USA"),
("George Washington University Hospital", "George Washington University Hospital, Washington DC, USA"),
("Inova Fairfax Hospital", "Inova Fairfax Hospital, Virginia, USA"),
("Howard County General Hospital", "Howard County General Hospital, Columbia, Maryland, USA"),
("MedStar Georgetown University Hospital", "MedStar Georgetown University Hospital, Washington DC, USA")],
"England, United Kingdom": [
("St Thomas' Hospital", "St Thomas' Hospital, London, England"),
("John Radcliffe Hospital", "John Radcliffe Hospital, Oxford, England"),
("Addenbrooke's Hospital", "Addenbrooke's Hospital, Cambridge, England"),
("Royal Berkshire Hospital", "Royal Berkshire Hospital, Reading, England"),
("Luton and Dunstable Hospital", "Luton and Dunstable University Hospital, Luton, England"),
("King's College Hospital", "King's College Hospital, London, England"),
("Churchill Hospital", "Churchill Hospital, Oxford, England"),
("Watford General Hospital", "Watford General Hospital, England"),
("Milton Keynes University Hospital", "Milton Keynes University Hospital, England"),
("Bedford Hospital", "Bedford Hospital, England")],
"Delhi, India": [
("AIIMS New Delhi", "AIIMS New Delhi, India"),
("Safdarjung Hospital", "Safdarjung Hospital, New Delhi, India"),
("Dr. Ram Manohar Lohia Hospital", "Dr Ram Manohar Lohia Hospital, New Delhi, India"),
("Maulana Azad Medical College", "Maulana Azad Medical College, Delhi, India"),
("Guru Teg Bahadur Hospital", "Guru Teg Bahadur Hospital, Delhi, India"),
("Sir Ganga Ram Hospital", "Sir Ganga Ram Hospital, Delhi, India"),
("Indraprastha Apollo Hospital", "Indraprastha Apollo Hospital, Delhi, India"),
("Holy Family Hospital", "Holy Family Hospital, Delhi, India"),
("Deen Dayal Upadhyay Hospital", "Deen Dayal Upadhyay Hospital, Delhi, India"),
("Hindu Rao Hospital", "Hindu Rao Hospital, Delhi, India")],
"Tokyo, Japan": [
("Tokyo Metropolitan Bokutoh Hospital", "Tokyo Metropolitan Bokutoh Hospital, Tokyo, Japan"),
("University of Tokyo Hospital", "The University of Tokyo Hospital, Tokyo, Japan"),
("St. Luke's International Hospital", "St. Luke's International Hospital, Tokyo, Japan"),
("Tokyo Metropolitan Hiroo Hospital", "Tokyo Metropolitan Hiroo Hospital, Tokyo, Japan"),
("Keio University Hospital", "Keio University Hospital, Tokyo, Japan"),
("National Center for Global Health and Medicine", "National Center for Global Health and Medicine, Tokyo, Japan"),
("Toranomon Hospital", "Toranomon Hospital, Tokyo, Japan"),
("Juntendo University Hospital", "Juntendo University Hospital, Tokyo, Japan"),
("Tokyo Medical University Hospital", "Tokyo Medical University Hospital, Tokyo, Japan"),
("Mitsui Memorial Hospital", "Mitsui Memorial Hospital, Tokyo, Japan")],
"Cape Town, South Africa": [
("Groote Schuur Hospital", "Groote Schuur Hospital, Cape Town, South Africa"),
("Red Cross War Memorial Children's Hospital", "Red Cross War Memorial Children's Hospital, Cape Town, South Africa"),
("New Somerset Hospital", "New Somerset Hospital, Cape Town, South Africa"),
("Tygerberg Hospital", "Tygerberg Hospital, Cape Town, South Africa"),
("Victoria Hospital", "Victoria Hospital, Wynberg, Cape Town, South Africa"),
("Mitchells Plain Hospital", "Mitchells Plain Hospital, Cape Town, South Africa"),
("Karl Bremer Hospital", "Karl Bremer Hospital, Cape Town, South Africa"),
("Mediclinic Cape Town", "Mediclinic Cape Town, South Africa"),
("Vincent Pallotti Hospital", "Vincent Pallotti Hospital, Cape Town, South Africa"),
("Netcare Christiaan Barnard Memorial Hospital", "Netcare Christiaan Barnard Memorial Hospital, Cape Town, South Africa")]
}

def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"resource-constrained-agent-tsp-benchmark/1.0"})
    with urllib.request.urlopen(req,timeout=90) as r: return json.load(r)

def tsp(matrix, nodes):
    start=nodes[0]; best=float("inf"); best_order=None
    for mid in itertools.permutations(nodes[1:]):
        order=(start,*mid,start); total=sum(matrix[a][b] for a,b in zip(order,order[1:]))
        if total<best: best,best_order=total,order
    return best,best_order

rng=random.Random(20260903); data={"source":{"geocoder":"Nominatim / OpenStreetMap","router":"OSRM public server, driving profile","retrieved":"2026-09-03","attribution":"OpenStreetMap contributors, ODbL 1.0"},"regions":{},"instances":[]}; benchmarks=[]
for region, requested in POOLS.items():
    places=[]
    for label,query in requested:
        url="https://nominatim.openstreetmap.org/search?"+urllib.parse.urlencode({"q":query,"format":"jsonv2","limit":1})
        found=get(url)
        if not found: raise RuntimeError(f"No result: {query}")
        x=found[0]; places.append({"name":label,"query":query,"lat":float(x["lat"]),"lon":float(x["lon"]),"osm_feature":f'{x["osm_type"]}/{x["osm_id"]}'})
        time.sleep(1.05)
    coords=";".join(f'{p["lon"]},{p["lat"]}' for p in places)
    routed=get(f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance")
    matrix=routed.get("distances")
    if routed.get("code")!="Ok" or not matrix or any(v is None for row in matrix for v in row): raise RuntimeError(f"Routing failed: {region}")
    data["regions"][region]={"places":places,"distances_meters":matrix}
    candidates=[c for size in range(4,8) for c in itertools.combinations(range(10),size)]
    rng.shuffle(candidates)
    for local_no, subset in enumerate(candidates[:200]):
        shift=local_no%len(subset); nodes=subset[shift:]+subset[:shift]
        distance,order=tsp(matrix,nodes); km=int((Decimal(str(distance))/Decimal(1000)).quantize(Decimal(1),rounding=ROUND_HALF_UP))
        names=[places[i]["name"] for i in nodes]; start=names[0]; visits=", ".join(names[1:])
        prompt=f"A public-health logistics vehicle starts at {start} and must visit {visits} exactly once each before returning to {start}. Using driving roads, what is the shortest possible total tour distance?"
        locations=[{"name":places[i]["name"],"latitude":places[i]["lat"],"longitude":places[i]["lon"]} for i in nodes]
        benchmarks.append({"prompt":prompt,"locations":locations,"answer":km,"geographic_region":region})
        data["instances"].append({"region":region,"nodes":list(nodes),"optimal_order":list(order),"optimal_meters":distance})
(ROOT/"benchmarks").mkdir(exist_ok=True)
(ROOT/"benchmarks"/"tsp.json").write_text(json.dumps(benchmarks,indent=2,ensure_ascii=False)+"\n")
(ROOT/"scripts"/"tsp_road_data.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
print(f"Generated {len(benchmarks)} benchmarks")
