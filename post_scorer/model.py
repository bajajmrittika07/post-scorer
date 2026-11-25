"""Regression: which features are linked to above-usual engagement, holding the others constant.

Ordinary least squares on log2(uplift) with robust (HC1) standard errors. Coefficients are turned
back into multipliers (2**coef), so 0.67 means "about a third less engagement than a comparable
text-only post", all else equal. This shows association, not cause.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

STYLE = ["heavy_caps", "has_exclamation", "has_nickname", "has_question", "has_call_to_action", "late_night", "long"]


def fit(df: pd.DataFrame, platform: str, min_year: str | None = None) -> pd.DataFrame:
    d = df[(df["platform"] == platform) & df["uplift"].notna() & (df["uplift"] > 0) & df["has_text"]].copy()
    if min_year:
        d = d[d["created_utc"] >= pd.Timestamp(min_year, tz="UTC")]
    d["y"] = np.log2(d["uplift"])
    d["fmt"] = d["format"].str.replace(r"[^A-Za-z]", "_", regex=True)
    d["long"] = d["n_words"] > 50
    d["log2_posts_that_day"] = np.log2(d["posts_that_day"])   # effect per doubling of that day's post count
    topic_cols = [c for c in d if c.startswith("topic_")]
    for c in topic_cols + STYLE:
        d[c] = d[c].astype(int)
    f = "y ~ C(fmt, Treatment('Text_only')) + " + " + ".join(topic_cols + STYLE) + " + log2_posts_that_day"
    if platform == "Truth Social":
        d["sent"] = d["src_sentiment_label"].fillna("neutral")
        d["sharp_language"] = (d["src_toxicity_score"] >= d["src_toxicity_score"].quantile(0.8)).astype(int)
        f += " + C(sent, Treatment('neutral')) + sharp_language"
    m = smf.ols(f, d).fit(cov_type="HC1")
    ci = m.conf_int()
    out = pd.DataFrame({"term": m.params.index, "multiplier": 2 ** m.params.values,
                        "lo": 2 ** ci[0].values, "hi": 2 ** ci[1].values, "p": m.pvalues.values})
    out = out[out["term"] != "Intercept"]
    out["platform"] = platform
    out["n"] = int(m.nobs)
    out["r2"] = round(m.rsquared, 3)
    return out
