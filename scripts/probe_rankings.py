"""Probe candidate FIFA ranking URLs to find a working free source."""
import urllib.request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

URLS = [
    "https://raw.githubusercontent.com/hericlibong/FifaRankingScraper/main/historicalmenranking/historicalmenranking/spiders/data.csv",
    "https://raw.githubusercontent.com/nicholasgasior/fifaRankings/master/ranking.csv",
    "https://raw.githubusercontent.com/kevinmey/fifaworldranking/master/rankings.csv",
    "https://raw.githubusercontent.com/stefanb/fifa-ranking/master/data/20221006-fifa-world-ranking.csv",
    "https://raw.githubusercontent.com/Soulforged/fifaRankings/master/data/2022/ranking.csv",
    "https://raw.githubusercontent.com/cnickels21/FIFA-Rankings/main/Data/FIFA_Rankings_1992_to_2022.csv",
    "https://raw.githubusercontent.com/cnickels21/FIFA-Rankings/main/Data/FIFA_Rankings.csv",
    "https://raw.githubusercontent.com/marcosg1928/FIFA-Ranking-Analysis/main/fifaRanking.csv",
    "https://raw.githubusercontent.com/robmarkcole/football-data/master/data/world_ranking.csv",
]

for url in URLS:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            chunk = r.read(500).decode("utf-8", errors="replace")
            print(f"OK ({r.status}): {url}")
            print(f"  First 300 chars: {chunk[:300]}")
    except Exception as e:
        print(f"FAIL: {url} -> {type(e).__name__}: {e}")
