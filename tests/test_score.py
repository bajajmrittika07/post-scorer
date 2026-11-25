"""Scoring: ±14-day median baseline, uplift, and the three exclusions."""
import numpy as np
import pandas as pd
from post_scorer.score import add_baseline


def daily(n_days=60, eng=100, start="2025-01-01 12:00"):
    t = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"platform": "Truth Social", "row_id": [f"p{i}" for i in range(n_days)], "created_utc": t,
                         "eng": eng, "is_repost": False, "is_reply": False, "is_deleted": False, "is_flagged": False})


def add(df, **row):
    base = {"platform": "Truth Social", "is_repost": False, "is_reply": False, "is_deleted": False, "is_flagged": False}
    row = base | row
    row["created_utc"] = pd.Timestamp(row["created_utc"], tz="UTC")
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def test_uplift_is_engagement_over_usual():
    df = add(daily(), row_id="star", created_utc="2025-01-30 18:00", eng=300)
    out = add_baseline(df).set_index("row_id")
    assert out.loc["p29", "usual"] == 100 and out.loc["p29", "uplift"] == 1.0
    assert out.loc["star", "uplift"] == 3.0


def test_reposts_and_replies_are_not_scored_or_in_baseline():
    df = daily()
    for i in range(40):   # a flood of zero-count reposts must not drag the baseline down
        df = add(df, row_id=f"rt{i}", created_utc=f"2025-01-30 0{i % 9}:00", eng=0, is_repost=True)
    out = add_baseline(df).set_index("row_id")
    assert out.loc["p29", "usual"] == 100
    assert out.filter(like="rt", axis=0)["uplift"].isna().all()


def test_deleted_and_flagged_are_excluded_and_kept_out_of_baseline():
    df = daily()
    for i in range(40):
        df = add(df, row_id=f"del{i}", created_utc=f"2025-01-29 0{i % 9}:00", eng=0, is_deleted=True)
    df = add(df, row_id="flag", created_utc="2025-01-29 12:00", eng=0, is_flagged=True)
    out = add_baseline(df).set_index("row_id")
    assert out.loc["p28", "usual"] == 100                       # 41 zero-count posts did not move the median
    assert (out.filter(like="del", axis=0)["exclude"] == "deleted").all()
    assert out.loc["flag", "exclude"] == "flagged" and np.isnan(out.loc["flag", "uplift"])


def test_tiny_engagement_excluded_but_low_engagement_kept():
    df = add(daily(), row_id="zero", created_utc="2025-01-30 18:00", eng=1)       # 1% of usual
    df = add(df, row_id="low", created_utc="2025-01-31 18:00", eng=3)              # 3% of usual
    out = add_baseline(df).set_index("row_id")
    assert out.loc["zero", "exclude"] == "tiny" and np.isnan(out.loc["zero", "uplift"])
    assert out.loc["low", "exclude"] == "" and out.loc["low", "uplift"] == 0.03


def test_too_few_posts_in_window_gives_no_score():
    df = add(daily(n_days=10), row_id="lonely", created_utc="2025-06-01 12:00", eng=100)
    out = add_baseline(df).set_index("row_id")
    assert np.isnan(out.loc["lonely", "uplift"])
