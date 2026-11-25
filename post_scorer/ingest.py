"""Load the Twitter archive and the Truth Social export into one schema.

One row per post. Columns:
  platform, row_id, created_utc, created_et, year, era, text,
  is_repost, is_reply, is_fan_quote, is_deleted, has_text, has_link, link_domain, format,
  likes, reposts, replies, eng (likes + reposts, comparable across platforms),
  eng_full (platform definition: Twitter likes+retweets; Truth Social likes+reposts+replies)
"""
import re
import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://(?:www\.)?([^/\s]+)", re.I)
VIDEO_DOMAINS = {"rumble.com", "youtube.com", "youtu.be", "rsbnetwork.com", "m.youtube.com"}

ERAS = [  # (label, start inclusive, end exclusive) in UTC
    ("2009–15 Pre-campaign", "2009-01-01", "2015-06-16"),
    ("2015–16 First campaign", "2015-06-16", "2017-01-20"),
    ("2017–21 First term", "2017-01-20", "2021-01-20"),
    ("2022–24 Truth Social, out of office", "2022-01-01", "2025-01-20"),
    ("2025–26 Second term", "2025-01-20", "2027-01-01"),
]


def fix_mojibake(t: str) -> str:
    """Some Truth Social texts were UTF-8 bytes read as Windows-1252/Latin-1 (e.g. 'â\x80\x94' for '—').
    Re-encode each character back to its byte, then decode as UTF-8. Leaves clean text untouched."""
    if "â" not in t and "Ã" not in t:
        return t
    out = bytearray()
    for ch in t:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            try:
                out += ch.encode("latin-1")
            except UnicodeEncodeError:
                out += ch.encode("utf-8")
    return out.decode("utf-8", errors="replace")


def _era(ts: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=ts.index, dtype="object")
    for label, a, b in ERAS:
        m = (ts >= pd.Timestamp(a, tz="UTC")) & (ts < pd.Timestamp(b, tz="UTC"))
        out[m] = label
    return out


def _domain(text: pd.Series) -> pd.Series:
    d = text.fillna("").str.extract(URL_RE)[0].str.lower()
    return d.str.replace(r"^www\.", "", regex=True)


def load_twitter(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.DataFrame({"platform": "Twitter", "row_id": "tw_" + raw.index.astype(str)})
    df["created_utc"] = pd.to_datetime(raw["date"], format="%m/%d/%Y %H:%M").dt.tz_localize("UTC")
    df["text"] = raw["text"].str.replace('^"+|"+$', "", regex=True).str.replace('""', '"')
    df["is_repost"] = (raw["isRetweet"] == "t") | df["text"].str.startswith("RT @")
    df["is_fan_quote"] = df["text"].str.match(r'^"?@\w+:')     # old-style manual quote of a fan ("@user: ...")
    df["is_reply"] = df["text"].str.match(r"^@\w") & ~df["is_fan_quote"]  # starts with @handle = reply/mention-led
    df["is_deleted"] = raw["isDeleted"] == "t"
    df["has_text"] = df["text"].str.strip() != ""
    df["link_domain"] = _domain(df["text"])
    df["has_link"] = df["link_domain"].notna()
    df["device"] = raw["device"]
    df["likes"] = raw["favorites"].astype(int)
    df["reposts"] = raw["retweets"].astype(int)
    df["replies"] = np.nan
    df["eng_full"] = df["likes"] + df["reposts"]
    # format: t.co hides whether a link is media or an article, so Twitter gets a coarser format
    fmt = np.where(df["is_repost"], "Repost",
          np.where(df["device"].isin(["Twitter Media Studio", "Media Studio", "Periscope", "Vine - Make a Scene"]), "Video",
          np.where(df["is_fan_quote"], "Quoted fan",
          np.where(df["has_link"], "Link or media", "Text only"))))
    df["format"] = fmt
    return df


def load_truth(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = pd.DataFrame({"platform": "Truth Social", "row_id": "ts_" + raw["id"].astype(str)})
    df["created_utc"] = pd.to_datetime(raw["created_at"], utc=True, format="mixed")
    df["text"] = raw["text"].fillna("").map(fix_mojibake)
    df["url"] = raw["url"]
    # is_retruth in the source is False for every row; reposts are identified from the "RT @" prefix instead
    df["is_repost"] = df["text"].str.match(r"^RT[:\s]*(@|https?://)")
    df["is_reply"] = df["text"].str.match(r"^@\w+\s*$")      # bare @handle posts
    df["is_fan_quote"] = False
    df["is_deleted"] = False
    df["has_text"] = df["text"].str.strip() != ""
    df["link_domain"] = _domain(df["text"])
    df["has_link"] = df["link_domain"].notna()
    df["likes"] = raw["like_count"]
    df["reposts"] = raw["retruth_count"]
    df["replies"] = raw["reply_count"]
    df["eng_full"] = df["likes"] + df["reposts"] + df["replies"]
    bare_link = df["text"].str.match(r"^\s*https?://\S+\s*$")
    is_video = df["link_domain"].isin(VIDEO_DOMAINS)
    is_truth_share = df["link_domain"].eq("truthsocial.com")
    fmt = np.select(
        [df["is_repost"], ~df["has_text"], is_video, is_truth_share, bare_link, df["has_link"]],
        ["Repost", "Image/video, no caption", "Video link", "Shared Truth post", "Article link only", "Text + article link"],
        default="Text only")
    df["format"] = fmt
    # keep the pre-made labels from the export for comparison, prefixed src_
    for c in ["topic", "rhetorical_device", "target_entity", "toxicity_score", "sentiment_label", "word_count",
              "cap_ratio", "all_caps_word_count", "exclamation_count", "is_policy_related"]:
        df["src_" + c] = raw[c]
    return df


def load_all(tw_path: str, ts_path: str) -> pd.DataFrame:
    df = pd.concat([load_twitter(tw_path), load_truth(ts_path)], ignore_index=True)
    df["eng"] = df["likes"] + df["reposts"]
    df["created_et"] = df["created_utc"].dt.tz_convert("America/New_York")
    df["year"] = df["created_utc"].dt.year
    df["era"] = _era(df["created_utc"])
    return df
