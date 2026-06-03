"""Check full date range of hericlibong FifaRankingScraper dataset."""
import urllib.request
import io
import csv
from collections import Counter

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

URL = "https://raw.githubusercontent.com/hericlibong/FifaRankingScraper/main/historicalmenranking/historicalmenranking/spiders/data.csv"

req = urllib.request.Request(URL, headers=HEADERS)
with urllib.request.urlopen(req, timeout=30) as r:
    content = r.read().decode("utf-8", errors="replace")

lines = content.strip().splitlines()
reader = csv.DictReader(io.StringIO(content))
rows = list(reader)
print(f"Total rows: {len(rows)}")
print(f"Columns: {reader.fieldnames}")
dates = sorted(set(r['date'] for r in rows))
print(f"Distinct dates ({len(dates)}): {dates[:10]} ... {dates[-10:]}")
print(f"Min date: {min(dates)}")
print(f"Max date: {max(dates)}")
# Check if any dates in range 2018-2022-11
oct_nov_2022 = [d for d in dates if d >= '2022-10-01' and d <= '2022-11-19']
print(f"Dates in Oct-Nov 2022 range: {oct_nov_2022}")
range_2018 = [d for d in dates if d >= '2018-01-01' and d <= '2018-12-31']
print(f"Dates in 2018: {range_2018}")
# Check distinct countries
countries = sorted(set(r['country'] for r in rows))
print(f"\nDistinct countries: {len(countries)}")
