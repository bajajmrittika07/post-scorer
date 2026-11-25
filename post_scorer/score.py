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


def add_baseline(df: pd.DataFrame, metric: str = "eng") -> pd.DataFrame:
    df = df.copy()
    df["is_original"] = ~df["is_repost"] & ~df["is_reply"]
    df["usual"] = np.nan
    df["window_n"] = 0
    for plat, g in df[df["is_original"]].groupby("platform"):
        g = g.sort_values("created_utc")
        s = pd.Series(g[metric].values, index=g["created_utc"].values)
        span = f"{2 * WINDOW_DAYS + 1}D"
        med = s.rolling(span, center=True, min_periods=1).median()
        cnt = s.rolling(span, center=True, min_periods=1).count()
        df.loc[g.index, "usual"] = med.values
        df.loc[g.index, "window_n"] = cnt.values
    ok = df["is_original"] & (df["window_n"] >= MIN_POSTS_IN_WINDOW) & (df["usual"] > 0)
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
