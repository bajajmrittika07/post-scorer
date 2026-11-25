"""Tagging: topic rules, style features and the posts-that-day count."""
import pandas as pd
from post_scorer.tag import tag


def frame(texts, times=None, platform="Truth Social"):
    times = times or ["2025-03-01 15:00"] * len(texts)
    utc = pd.to_datetime(pd.Series(times)).dt.tz_localize("UTC")
    return pd.DataFrame({"platform": platform, "row_id": [f"r{i}" for i in range(len(texts))], "text": texts,
                         "has_text": [t.strip() != "" for t in texts], "created_utc": utc,
                         "created_et": utc.dt.tz_convert("America/New_York")})


def test_topic_rules_and_priority():
    df = tag(frame(["I give my Complete and Total Endorsement to Jane Doe. Vote!",
                    "As President of the United States, I am proud",
                    "Great meeting with President Xi of China",
                    "We will finish the border wall",
                    "",
                    "Lovely weather today"]))
    assert df.loc[0, "topic"] == "Endorsements"                       # endorsement outranks elections
    assert df.loc[0, "topic_elections_campaign"]                       # but both are flagged (multi-label)
    assert not df.loc[1, "topic_foreign_policy_military"]              # the "President of the United States" fix
    assert df.loc[2, "topic_foreign_policy_military"]
    assert df.loc[3, "topic"] == "Immigration & border"
    assert df.loc[4, "topic"] == "No text (media only)"
    assert df.loc[5, "topic"] == "Other / unclear"


def test_style_features():
    df = tag(frame(["Crooked Hillary is a TOTAL DISASTER for our COUNTRY!", "a quiet post with some words?"]))
    assert df.loc[0, "has_nickname"] and df.loc[0, "has_exclamation"] and df.loc[0, "heavy_caps"]
    # words are counted as 2+ letters, so the single-letter "a" is not counted
    assert not df.loc[1, "heavy_caps"] and df.loc[1, "has_question"] and df.loc[1, "n_words"] == 5


def test_posts_that_day_counts_by_eastern_day_and_platform():
    times = ["2021-01-01 03:00", "2021-01-01 04:30",       # 31 Dec in US Eastern
             "2021-01-01 15:00", "2021-01-01 16:00", "2021-01-01 17:00"]   # 1 Jan in US Eastern
    df = tag(frame(["a"] * 5, times))
    assert df["posts_that_day"].tolist() == [2, 2, 3, 3, 3]
    other = frame(["b"], ["2021-01-01 15:00"], platform="Twitter")
    both = tag(pd.concat([frame(["a"] * 3, times[2:]), other], ignore_index=True))
    assert both.loc[both.platform == "Twitter", "posts_that_day"].iloc[0] == 1   # platforms counted separately
