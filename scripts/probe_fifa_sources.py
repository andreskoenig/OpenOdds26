"""Probe candidate live FIFA-ranking CSV sources: status, header, date range."""
import csv, io, re, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
CANDIDATES = [
    "https://raw.githubusercontent.com/cnc8/fifa-world-ranking/master/fifa_ranking.csv",
    "https://raw.githubusercontent.com/cnc8/fifa-world-ranking/main/fifa_ranking.csv",
    "https://raw.githubusercontent.com/cnc8/fifa-world-ranking/master/data/fifa_ranking.csv",
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/data/ranking_fifa_historical.csv",
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/main/data/ranking_fifa_historical.csv",
]
for url in CANDIDATES:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"[{type(e).__name__}] {url}")
        continue
    rows = list(csv.DictReader(io.StringIO(raw)))
    cols = list(rows[0].keys()) if rows else []
    # find a date-like column
    dcol = next((c for c in cols if "date" in c.lower()), None)
    drange = ""
    if dcol:
        ds = [r[dcol].strip() for r in rows if r.get(dcol, "").strip()]
        if ds:
            drange = f" | {dcol} range {min(ds)} .. {max(ds)}"
    print(f"[OK rows={len(rows)}] {url}")
    print(f"   cols={cols}{drange}")
