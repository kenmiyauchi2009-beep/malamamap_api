"""GBIF からハワイ州の植物種リスト（出現>=5件）を作り hawaii_plants.csv に保存する。"""
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://api.gbif.org/v1/occurrence/search"
GADM_HAWAII = "USA.12_1"
PLANTAE = 6
MIN_OCCURRENCES = 5
OUT_CSV = "hawaii_plants.csv"


def get_species_keys() -> dict:
    """ハワイの植物の speciesKey -> 出現件数 を全件取得。"""
    counts = {}
    offset, page = 0, 1000
    while True:
        url = (f"{BASE}?gadmGid={GADM_HAWAII}&taxonKey={PLANTAE}&limit=0"
               f"&facet=speciesKey&facetLimit={page}&facetOffset={offset}")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        buckets = data["facets"][0]["counts"] if data.get("facets") else []
        if not buckets:
            break
        for b in buckets:
            counts[b["name"]] = b["count"]
        offset += page
        if len(buckets) < page:
            break
    return counts


def resolve_name(species_key: str):
    """speciesKey を学名（二名法）に解決。種ランク・植物界のみ返す。"""
    url = f"https://api.gbif.org/v1/species/{species_key}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
    except Exception:
        return None
    if d.get("rank") != "SPECIES" or d.get("kingdom") != "Plantae":
        return None
    name = d.get("canonicalName") or d.get("species")
    return name.strip() if name else None


def main():
    counts = get_species_keys()
    keys = [k for k, c in counts.items() if c >= MIN_OCCURRENCES]
    print(f"出現>={MIN_OCCURRENCES}件の speciesKey: {len(keys)} 件を学名に解決中...")

    with ThreadPoolExecutor(max_workers=16) as ex:
        names = list(ex.map(resolve_name, keys))

    species = sorted({n for n in names if n})
    print(f"確定した植物種: {len(species)} 種")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fd:
        w = csv.writer(fd)
        w.writerow(["species"])
        for s in species:
            w.writerow([s])
    print(f"保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
