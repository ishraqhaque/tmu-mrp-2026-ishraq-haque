"""One-off diagnostic: probe which FMP endpoints + symbols this key can access."""
import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
key = os.getenv("FMP_API_KEY")
print("Key loaded:", (key[:4] + "..." + key[-3:]) if key else None)

tests = [
    ("v3 legacy  AAPL", "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL", {}),
    ("stable EOD AAPL", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "AAPL"}),
    ("stable EOD ^GSPC", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "^GSPC"}),
    ("stable EOD BTCUSD", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "BTCUSD"}),
    ("stable EOD GCUSD", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "GCUSD"}),
    ("stable EOD ^VIX", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "^VIX"}),
    ("stable EOD ^TNX", "https://financialmodelingprep.com/stable/historical-price-eod/full", {"symbol": "^TNX"}),
]
for label, url, params in tests:
    p = {**params, "apikey": key, "from": "2024-01-01", "to": "2024-01-10"}
    try:
        r = requests.get(url, params=p, timeout=20)
        body = r.text[:160].replace("\n", " ")
        n = ""
        if r.status_code == 200:
            try:
                j = r.json()
                if isinstance(j, list):
                    n = f" rows={len(j)}"
                elif isinstance(j, dict) and "historical" in j:
                    n = f" rows={len(j['historical'])}"
            except Exception:
                pass
        print(f"  [{r.status_code}] {label}{n}  :: {body}")
    except Exception as e:
        print(f"  [ERR] {label} :: {e}")
