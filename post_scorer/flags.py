"""Plain-English insight flags, in the style of the Knox Index profile flags.

Each flag is generated from the scored data, never typed by hand, so the sentence and the number
always match. A flag fires only when the group is big enough (n >= MIN_N) and the effect is clear
(at least 15% above or below his usual).
"""
import pandas as pd

MIN_N = 100
THRESHOLD = 0.15


def _fmt_x(v: float) -> str:
    return f"{v:.2f}x"


def make_flags(df: pd.DataFrame, group_col: str, label: str, platform: str, era_label: str | None = None) -> list[dict]:
    d = df[(df["platform"] == platform) & df["uplift"].notna()]
    if era_label:
        d = d[d["era"] == era_label]
    flags = []
    for key, g in d.groupby(group_col):
        if len(g) < MIN_N or key in ("Other / unclear",):
            continue
        m = g["uplift"].median()
        if abs(m - 1) < THRESHOLD:
            continue
        direction = "above" if m > 1 else "below"
        flags.append({
            "platform": platform, "era": era_label or "all", "dimension": label, "group": str(key),
            "median_uplift": round(float(m), 3), "n": int(len(g)),
            "level": "strong" if m >= 1.3 or m <= 0.75 else "moderate",
            "text": f"{key}: {_fmt_x(m)} his usual engagement on {platform}"
                    f"{' (' + era_label + ')' if era_label else ''} — {direction} his typical post, based on {len(g):,} posts.",
        })
    return sorted(flags, key=lambda f: -abs(f["median_uplift"] - 1))


def feature_flags(df: pd.DataFrame, features: dict, platform: str) -> list[dict]:
    """Binary style features: compare median uplift WITH vs WITHOUT the feature."""
    d = df[(df["platform"] == platform) & df["uplift"].notna() & df["has_text"]]
    out = []
    for col, name in features.items():
        w, wo = d[d[col]], d[~d[col]]
        if len(w) < MIN_N:
            continue
        ratio = w["uplift"].median() / wo["uplift"].median()
        if abs(ratio - 1) < THRESHOLD:
            continue
        pct = (ratio - 1) * 100
        out.append({
            "platform": platform, "dimension": "style", "group": name, "ratio": round(float(ratio), 3), "n": int(len(w)),
            "text": f"Posts with {name.lower()} do {abs(pct):.0f}% {'better' if pct > 0 else 'worse'} than posts without on {platform} ({len(w):,} posts).",
        })
    return sorted(out, key=lambda f: -abs(f["ratio"] - 1))
