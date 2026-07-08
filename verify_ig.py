"""
verify_ig.py — Test IG API credentials and data access.

Usage:
    python verify_ig.py

Checks:
    1. Authentication (POST /session, v3) — OAuth 2.0 Bearer token
    2. Market discovery (GET /categories, v1) — list available categories
    3. Price history (GET /prices/{epic}, v3) — OHLCV data retrieval

Related docs:
    https://labs.ig.com/rest-trading-api-reference.html
"""

import httpx

API_KEY = "bf86c2b12a056f19528c11b49136f28ddbe5f149"
USERNAME = "koatekpai_demo"
PASSWORD = "y6Ap#CpFabK#jFfM"

# Use demo API for non-live accounts; change to api.ig.com for live
BASE = "https://demo-api.ig.com/gateway/deal"

print("=" * 60)
print("Step 1: Authentication (POST /session, v3)")
print("=" * 60)

resp = httpx.post(
    f"{BASE}/session",
    headers={
        "X-IG-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "VERSION": "3",
    },
    json={"identifier": USERNAME, "password": PASSWORD},
    timeout=15,
)
print(f"  HTTP {resp.status_code}")

if resp.status_code != 200:
    print(f"  FAILED: {resp.text}")
    exit(1)

session = resp.json()
token = session["oauthToken"]["access_token"]
account_id = session["accountId"]
lightstreamer = session.get("lightstreamerEndpoint", "?")

print(f"  Account ID:     {account_id}")
print(f"  Client ID:      {session.get('clientId', '?')}")
print(f"  Lightstreamer:  {lightstreamer}")
print(f"  Access token:   {token[:40]}...")
print()

# Common auth headers for all subsequent calls (OAuth 2.0 Bearer)
AUTH = {
    "X-IG-API-KEY": API_KEY,
    "Authorization": f"Bearer {token}",
    "IG-ACCOUNT-ID": account_id,
    "VERSION": "1",          # most endpoints use v1
}

print("=" * 60)
print("Step 2: List market categories (GET /categories, v1)")
print("=" * 60)

r = httpx.get(f"{BASE}/categories", headers=AUTH, timeout=15)
print(f"  HTTP {r.status_code}")

if r.status_code == 200:
    categories = r.json().get("categories", [])
    print(f"  Categories: {len(categories)}")
    for c in categories:
        code = c.get("code", "?")
        tradeable = "" if c.get("nonTradeable") else " (tradeable)"
        print(f"    - {code}{tradeable}")
else:
    print(f"  FAILED: {r.text[:500]}")

print()

print("=" * 60)
print("Step 3: Fetch price history (GET /prices/{epic}, v3)")
print("=" * 60)

# Search for a liquid index to fetch prices for
r = httpx.get(
    f"{BASE}/markets",
    headers=AUTH | {"VERSION": "1"},        # /markets works at v1
    params={"searchTerm": "US 500"},
    timeout=15,
)

if r.status_code == 200:
    markets = r.json().get("markets", [])
    if markets:
        epic = markets[0].get("epic", "")
        name = markets[0].get("instrumentName", "")
        print(f"  Market:  {name} ({epic})")

        # Price history uses v3 API
        r2 = httpx.get(
            f"{BASE}/prices/{epic}",
            headers=AUTH | {"VERSION": "3"},
            params={"resolution": "DAY", "max": 5},
            timeout=15,
        )
        print(f"  HTTP {r2.status_code}")

        if r2.status_code == 200:
            prices = r2.json().get("prices", [])
            print(f"  Points:  {len(prices)}")
            for p in prices:
                close = p.get("closePrice", {})
                dt = p.get("snapshotTime", "?")
                bid = close.get("bid", "?")
                ask = close.get("ask", "?")
                bid_str = f"{bid:>10.2f}" if isinstance(bid, (int, float)) else str(bid)
                ask_str = f"{ask:>10.2f}" if isinstance(ask, (int, float)) else str(ask)
                print(f"    {str(dt):22s}  bid={bid_str:>10s}  ask={ask_str:>10s}")
        else:
            print(f"  FAILED: {r2.text[:500]}")
    else:
        print("  No markets found for 'US 500'")
else:
    print(f"  Market search failed: {r.text[:500]}")

print()
print("Done. IG API credentials verified successfully.")
