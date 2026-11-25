"""Ingest: both exports load into one schema, with the repost flag, encoding and format rules applied.
All data here is made up; the real CSVs are never used in tests."""
import pandas as pd
from post_scorer.ingest import fix_mojibake, load_twitter, load_truth, load_all

TS_COLS = ["id", "created_at", "text", "url", "retruth_count", "like_count", "reply_count", "topic",
           "rhetorical_device", "target_entity", "toxicity_score", "sentiment_label", "word_count",
           "cap_ratio", "all_caps_word_count", "exclamation_count", "is_policy_related"]


def write_twitter(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["id", "text", "isRetweet", "isDeleted", "device", "favorites", "retweets", "date", "isFlagged"])
    p = tmp_path / "tw.csv"; df.to_csv(p, index=False); return str(p)


def write_truth(tmp_path, texts):
    rows = [{c: None for c in TS_COLS} | {"id": 1000 + i, "created_at": f"2025-03-0{1 + i % 9} 15:00:00+00:00",
             "text": t, "url": "u", "retruth_count": 10, "like_count": 100, "reply_count": 5} for i, t in enumerate(texts)]
    p = tmp_path / "ts.csv"; pd.DataFrame(rows, columns=TS_COLS).to_csv(p, index=False); return str(p)


def test_fix_mojibake_repairs_and_leaves_clean_text_alone():
    clean = "It’s a great day — truly"
    broken = clean.encode("utf-8").decode("cp1252")
    assert broken != clean
    assert fix_mojibake(broken) == clean
    assert fix_mojibake("Plain ASCII text!") == "Plain ASCII text!"


def test_twitter_flags_and_formats(tmp_path):
    rows = [
        ["1", "Make America Great Again!", "f", "f", "Twitter for Android", "100", "50", "06/20/2016 14:00", "f"],
        ["2", "RT @someone: great rally", "t", "f", "Twitter for iPhone", "0", "900", "06/20/2016 15:00", "f"],
        ["3", '"@fan123: You are the best!" Thank you', "f", "f", "Twitter for Android", "80", "10", "06/20/2016 16:00", "f"],
        ["4", "@reporter you are wrong", "f", "f", "Twitter for Android", "40", "5", "06/20/2016 17:00", "f"],
        ["5", "Read this https://t.co/abc", "f", "t", "Twitter for iPhone", "0", "0", "06/20/2016 18:00", "f"],
        ["6", "Rigged election claim", "f", "f", "Twitter for iPhone", "0", "0", "11/10/2020 12:00", "t"],
    ]
    # row_id is the row position (tw_0, tw_1, …), not the id column: the real Twitter ids are corrupted
    df = load_twitter(write_twitter(tmp_path, rows)).set_index("row_id")
    assert df.loc["tw_0", "format"] == "Text only"
    assert df.loc["tw_1", "is_repost"] and df.loc["tw_1", "format"] == "Repost"
    assert df.loc["tw_2", "is_fan_quote"] and not df.loc["tw_2", "is_reply"] and df.loc["tw_2", "format"] == "Quoted fan"
    assert df.loc["tw_3", "is_reply"] and not df.loc["tw_3", "is_fan_quote"]
    assert df.loc["tw_4", "is_deleted"] and df.loc["tw_4", "format"] == "Link or media"
    assert df.loc["tw_5", "is_flagged"] and not df.loc["tw_0", "is_flagged"]
    assert df.loc["tw_0", "eng_full"] == 150


def test_truth_social_reposts_and_formats(tmp_path):
    texts = ["RT @someone great post", "RT: https://truthsocial.com/@x/1", "", "https://rumble.com/v123",
             "https://truthsocial.com/@x/999", "https://www.foxnews.com/story", "Must read! https://www.foxnews.com/story",
             "The economy is booming!"]
    df = load_truth(write_truth(tmp_path, texts))
    fmt = df["format"].tolist()
    assert fmt == ["Repost", "Repost", "Image/video, no caption", "Video link", "Shared Truth post",
                   "Article link only", "Text + article link", "Text only"]
    assert df["link_domain"].iloc[5] == "foxnews.com"        # www. stripped
    assert not df["is_deleted"].any() and not df["is_flagged"].any()


def test_load_all_adds_engagement_eras_and_eastern_time(tmp_path):
    tw = write_twitter(tmp_path, [["1", "Hello", "f", "f", "Twitter for Android", "10", "5", "01/21/2017 03:00", "f"]])
    ts = write_truth(tmp_path, ["Hello again"])
    df = load_all(tw, ts)
    t = df[df.platform == "Twitter"].iloc[0]
    assert t["eng"] == 15
    assert t["created_et"].day == 20 and t["created_et"].hour == 22      # 03:00 UTC on the 21st = 22:00 ET on the 20th
    assert t["era"] == "2017–21 First term"
    assert df[df.platform == "Truth Social"].iloc[0]["era"] == "2025–26 Second term"
