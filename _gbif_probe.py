"""ハワイ州の植物が GBIF に何種あるか、出現件数の分布を調べる。"""
import json
import urllib.request

BASE = "https://api.gbif.org/v1/occurrence/search"
GADM_HAWAII = "USA.12_1"
PLANTAE = 6

species_counts = {}  # speciesKey -> occurrence count
offset = 0
page = 1000
while True:
    url = (f"{BASE}?gadmGid={GADM_HAWAII}&taxonKey={PLANTAE}&limit=0"
           f"&facet=speciesKey&facetLimit={page}&facetOffset={offset}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    buckets = data["facets"][0]["counts"] if data.get("facets") else []
    if not buckets:
        break
    for b in buckets:
        species_counts[b["name"]] = b["count"]
    offset += page
    if len(buckets) < page:
        break

total = len(species_counts)
print(f"GBIF上のハワイ植物 distinct species: {total}")
for thr in (1, 3, 5, 10, 20, 50):
    n = sum(1 for c in species_counts.values() if c >= thr)
    print(f"  出現>={thr:>3}件の種数: {n}")
