"""Hand-collected follower snapshot for senior US politicians (comparison for the report).

Input: a CSV collected by hand from public profiles (kept private, never published):
    name,party,role,platform,handle,followers,checked_on
Followers are typed as the profile shows them ("111.7M", "44.1k", "3M", "302k").
A blank follower cell means the person has no account on that platform.
Only these platforms were collected: X, Instagram, Truth Social (TikTok was not checked).
The export keeps names, party, role, platform, count and date; handles are left out.
"""
import io
import pandas as pd

PLATFORMS = ["X", "Instagram", "Truth Social"]
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def parse_count(s: str) -> float | None:
    s = str(s).strip().replace(",", "")
    if not s:
        return None
    unit = s[-1].lower()
    if unit in _MULT:
        return float(s[:-1]) * _MULT[unit]
    return float(s)


def load_peers(path: str) -> pd.DataFrame:
    lines = open(path, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("name,"))      # skip any stray title line
    d = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str, keep_default_na=False)
    d = d[d["platform"].isin(PLATFORMS)].copy()
    d["followers_n"] = d["followers"].map(parse_count)
    d = d[d["followers_n"].notna()]                                             # blank = no account
    return d


def export_peers(path: str) -> list[dict]:
    d = load_peers(path)
    return [{"name": r.name, "party": r.party, "role": r.role, "platform": r.platform,
             "followers": int(r.followers_n), "shown_as": r.followers, "checked_on": r.checked_on}
            for r in d.itertuples()]
