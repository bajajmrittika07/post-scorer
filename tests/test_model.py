"""Model: the regression recovers effects planted in made-up data."""
import numpy as np
import pandas as pd
from post_scorer.model import fit, STYLE


def test_regression_recovers_planted_effects():
    rng = np.random.default_rng(0)
    n = 4000
    link = rng.random(n) < 0.4
    day = rng.integers(1, 64, n)
    d = pd.DataFrame({"platform": "Twitter", "has_text": True,
                      "created_utc": pd.Timestamp("2018-01-01", tz="UTC"),
                      "format": np.where(link, "Link or media", "Text only"),
                      "n_words": rng.integers(5, 80, n), "posts_that_day": day,
                      "topic_example": rng.random(n) < 0.3})
    for c in STYLE:
        if c != "long":
            d[c] = rng.random(n) < 0.3
    # planted truth: a link multiplies engagement by 0.7; each doubling of posts that day by 0.85; the topic has no effect
    log_u = np.log2(0.7) * link + np.log2(0.85) * np.log2(day) + rng.normal(0, 0.3, n)
    d["uplift"] = 2 ** log_u
    m = fit(d, "Twitter").set_index("term")
    link_term = [t for t in m.index if "Link_or_media" in t][0]
    assert abs(m.loc[link_term, "multiplier"] - 0.7) < 0.02
    assert abs(m.loc["log2_posts_that_day", "multiplier"] - 0.85) < 0.01
    assert abs(m.loc["topic_example", "multiplier"] - 1) < 0.03                  # no planted effect → close to 1
