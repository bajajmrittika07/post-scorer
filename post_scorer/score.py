"""Baseline and uplift, following the Knox Index method.

usual engagement = median engagement of the account's own ORIGINAL posts (no reposts, no replies)
on the same platform within 14 days either side of the post.
uplift = post engagement / usual engagement  (1.0 = a typical post at that time; 3.0 = three times usual).

Why: Trump's audience grew from thousands to tens of millions, so raw likes mostly measure *when*
a post was made. Uplift measures how a post did against what he normally got at the time.
"""
import numpy as np
import pandas as pd

WINDOW_DAYS = 14
MIN_POSTS_IN_WINDOW = 5
TINY_SHARE = 0.02   # engagement under 2% of usual = counts broken or switched off, not a real reaction

# Exclusions (logged in LOG.md). A post can be excluded for one reason only, checked in this order:
#   deleted  - deleted tweets: engagement froze at the moment of deletion
#   flagged  - tweets Twitter labelled in 2020; engagement was switched off on some of them
#   tiny     - engagement under 2% of usual (zeros and near-zeros; e.g. 19 mid-2016 link tweets that may
#              have been promoted-only ads - a guess, not a fact)
# Deleted and flagged posts are also left out of the baseline itself, so frozen counts don't pull "usual" down.
# Repeated texts are kept: they are genuine re-posts of the same message.


def add_baseline(df: pd.DataFrame, metric: str = "eng") -> pd.DataFrame:
    df = df.copy()
    df["is_original"] = ~df["is_repost"] & ~df["is_reply"]
    df["exclude"] = ""
    df.loc[df["is_original"] & df["is_deleted"], "exclude"] = "deleted"
    df.loc[df["is_original"] & df["is_flagged"] & (df["exclude"] == ""), "exclude"] = "flagged"
    base = df["is_original"] & (df["exclude"] == "")
    df["usual"] = np.nan
    df["window_n"] = 0
    for plat, g in df[base].groupby("platform"):
        g = g.sort_values("created_utc")
        s = pd.Series(g[metric].values, index=g["created_utc"].values)
        span = f"{2 * WINDOW_DAYS + 1}D"
        med = s.rolling(span, center=True, min_periods=1).median()
        cnt = s.rolling(span, center=True, min_periods=1).count()
        df.loc[g.index, "usual"] = med.values
        df.loc[g.index, "window_n"] = cnt.values
    ok = base & (df["window_n"] >= MIN_POSTS_IN_WINDOW) & (df["usual"] > 0)
    tiny = ok & (df[metric] < TINY_SHARE * df["usual"])
    df.loc[tiny, "exclude"] = "tiny"
    ok = ok & ~tiny
    df["uplift"] = np.where(ok, df[metric] / df["usual"], np.nan)
    df["log_uplift"] = np.log2(df["uplift"].where(df["uplift"] > 0))
    return df


def summarise(df: pd.DataFrame, by, min_n: int = 30) -> pd.DataFrame:
    """Median uplift, share of posts, and over-representation in the top 10% of posts."""
    d = df[df["uplift"].notna()].copy()
    d["top10"] = d.groupby("platform")["uplift"].transform(lambda u: u >= u.quantile(0.9))
    g = d.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "median_uplift": g["uplift"].median(),
        "share_beating_usual": g["uplift"].apply(lambda u: (u > 1).mean()),
        "share_in_top10": g["top10"].mean(),
        "median_eng": g["eng"].median(),
    })
    return out[out["n"] >= min_n].reset_index()


def topic_lift(df: pd.DataFrame, topic_cols, group: str = "platform", min_n: int = 30) -> pd.DataFrame:
    """Multi-label view: for each topic flag, median uplift of posts WITH it vs WITHOUT it."""
    d = df[df["uplift"].notna()]
    rows = []
    for gkey, gd in d.groupby(group):
        top_cut = gd["uplift"].quantile(0.9)
        for col in topic_cols:
            w = gd[gd[col]]
            if len(w) < min_n:
                continue
            rows.append({group: gkey, "feature": col, "n": len(w),
                         "share_of_posts": len(w) / len(gd),
                         "share_of_top10": (gd[gd["uplift"] >= top_cut][col]).mean(),
                         "median_uplift_with": w["uplift"].median(),
                         "median_uplift_without": gd[~gd[col]]["uplift"].median()})
    return pd.DataFrame(rows)
