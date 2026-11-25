"""Write the aggregated JSON files the web report reads. No raw dataset is exported:
only summary tables plus a curated, truncated set of example posts for the explorer."""
import json
import re
import numpy as np
import pandas as pd

from .score import summarise, topic_lift
from .flags import make_flags, feature_flags
from .model import fit
from .tag import TOPIC_RULES

TOPIC_NAMES = {"topic_" + re.sub(r"\W+", "_", k.lower()).strip("_"): k for k in TOPIC_RULES}
ANALYSIS_START_TWITTER = pd.Timestamp("2015-06-16", tz="UTC")  # campaign launch. Before it, engagement was recorded years later and is inflated by rediscovered tweets

TERM_LABELS = {
    "Article_link_only": "Article link, no text", "Shared_Truth_post": "Sharing another Truth post",
    "Text___article_link": "Text + article link", "Video_link": "Video link", "Link_or_media": "Link or media",
    "Quoted_fan": "Quoting a fan", "Video": "Native video",
    "heavy_caps": "CAPITALS (¼ of words or more)", "has_exclamation": "Exclamation mark",
    "has_nickname": "Nickname for an opponent", "has_question": "Asks a question",
    "has_call_to_action": "Call to action (vote, watch, join…)", "late_night": "Posted midnight–6am ET",
    "long": "Long post (51+ words)", "sharp_language": "Sharpest-language fifth of posts",
    "C(sent, Treatment('neutral'))[T.negative]": "Negative tone (vs neutral)",
    "C(sent, Treatment('neutral'))[T.positive]": "Positive tone (vs neutral)",
}


def _label(term: str) -> tuple[str, str]:
    m = re.match(r"C\(fmt.*\)\[T\.(.+)\]", term)
    if m:
        return "Format (vs text only)", TERM_LABELS.get(m.group(1), m.group(1))
    if term.startswith("topic_"):
        return "Topic", TOPIC_NAMES.get(term, term)
    if term.startswith("C(sent"):
        return "Tone", TERM_LABELS.get(term, term)
    return "Style", TERM_LABELS.get(term, term)


def _r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def era_short(df):
    return np.where(df["platform"] == "Twitter",
                    np.where(df["created_utc"] < pd.Timestamp("2015-06-16", tz="UTC"), "Twitter 2009–15", np.where(df["created_utc"] < pd.Timestamp("2017-01-20", tz="UTC"), "Twitter 2015–16", "Twitter 2017–21")),
                    np.where(df["created_utc"] < pd.Timestamp("2025-01-20", tz="UTC"), "Truth Social 2022–24", "Truth Social 2025–26"))


def build(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["era_s"] = era_short(df)
    early = (df["platform"] == "Twitter") & (df["created_utc"] < ANALYSIS_START_TWITTER)
    full = df                      # volume and timeline use everything
    df = df.copy()
    df.loc[early, "uplift"] = np.nan   # uplift-based analysis: Twitter 2015-21 + Truth Social
    out = {}

    # headline numbers
    out["headline"] = {
        "total_posts": int(len(df)), "twitter_posts": int((df.platform == "Twitter").sum()),
        "truth_posts": int((df.platform == "Truth Social").sum()),
        "scored_posts": int(df["uplift"].notna().sum()),
        "first_post": str(df["created_utc"].min().date()), "last_post": str(df["created_utc"].max().date()),
    }

    # monthly timeline
    m = full.assign(month=df["created_utc"].dt.strftime("%Y-%m"))
    tl = m.groupby(["platform", "month"]).agg(posts=("row_id", "size"), reposts=("is_repost", "sum")).reset_index()
    eng = m[m["is_original"]].groupby(["platform", "month"])["eng"].median().rename("median_eng").reset_index()
    tl = tl.merge(eng, on=["platform", "month"], how="left")
    out["timeline"] = [{k: (_r(v, 0) if isinstance(v, float) else v) for k, v in r.items()} for r in tl.to_dict("records")]

    # era scorecard (Twitter 2016 vs Truth Social 2026 etc.)
    rows = []
    for e, g in full.groupby("era_s"):
        o = g[g["is_original"]]
        days = (g["created_utc"].max() - g["created_utc"].min()).days + 1
        sc = df.loc[o.index][df.loc[o.index, "uplift"].notna()]
        best_hour = sc.groupby("hour_et")["uplift"].median()
        best_hour = best_hour[sc.groupby("hour_et").size() >= 50]
        rows.append({
            "era": e, "posts": int(len(g)), "posts_per_day": _r(len(g) / days, 1),
            "repost_share": _r(g["is_repost"].mean()), "median_eng": _r(o["eng"].median(), 0),
            "median_likes": _r(o["likes"].median(), 0),
            "text_only_share": _r((o["format"] == "Text only").mean()), "link_share": _r(o["has_link"].mean()),
            "no_text_share": _r((~o["has_text"]).mean()),
            "caps_share": _r(o["heavy_caps"].mean()), "nickname_share": _r(o["has_nickname"].mean()),
            "exclamation_share": _r(o["has_exclamation"].mean()), "median_words": _r(o.loc[o.has_text, "n_words"].median(), 0),
            "long_share": _r((o["n_words"] > 50).mean()), "late_night_share": _r(o["late_night"].mean()),
            "best_hour_et": int(best_hour.idxmax()) if len(best_hour) else None,
        })
    order = ["Twitter 2009–15", "Twitter 2015–16", "Twitter 2017–21", "Truth Social 2022–24", "Truth Social 2025–26"]
    out["eras"] = sorted(rows, key=lambda r: order.index(r["era"]))

    tw20 = full[(full.platform == "Twitter") & (full.created_utc >= pd.Timestamp("2020-01-01", tz="UTC")) & (full.created_utc < pd.Timestamp("2020-06-01", tz="UTC"))]
    ts25 = full[(full.platform == "Truth Social") & (full.created_utc >= pd.Timestamp("2025-01-20", tz="UTC"))]
    ts_months = (ts25.created_utc.max() - ts25.created_utc.min()).days / 30.44
    out["volume_compare"] = {"trump_jan_may_2020_per_month": _r(len(tw20) / 5, 0),
                             "trump_ts_2025_26_per_month": _r(len(ts25) / ts_months, 0)}

    # format uplift
    fs = summarise(df, ["platform", "format"])
    out["formats"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in fs.to_dict("records")]

    # topic share vs share of top 10% (Knox "share vs baseline"), multi-label, by platform
    tcols = [c for c in df if c.startswith("topic_")]
    tl_ = topic_lift(df[df["uplift"].notna()], tcols, group="era_s")
    tl_["topic"] = tl_["feature"].map(lambda c: _label(c)[1])
    out["topics"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items() if k != "feature"} for r in tl_.to_dict("records")]

    # style features by era (ratio with / without)
    feats = {"has_nickname": "Nickname for an opponent", "heavy_caps": "CAPITALS", "has_exclamation": "Exclamation mark",
             "has_link": "Contains a link", "has_question": "Asks a question", "has_call_to_action": "Call to action",
             "late_night": "Posted midnight–6am ET"}
    st = topic_lift(df[df["has_text"]], list(feats), group="era_s")
    st["feature"] = st["feature"].map(feats)
    st["ratio"] = st["median_uplift_with"] / st["median_uplift_without"]
    out["style"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in st.to_dict("records")]

    # length
    ln = summarise(df[df["has_text"]], ["era_s", "length"], min_n=50)
    ln["length"] = ln["length"].astype(str)
    out["length"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in ln.to_dict("records")]

    # hour of day
    hr = summarise(df, ["platform", "hour_et"], min_n=50)
    out["hours"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in hr.to_dict("records")]

    # Truth Social source-label cross-check (sentiment, rhetoric)
    ts = df[(df.platform == "Truth Social") & df["uplift"].notna()].copy()
    ts["rhet"] = ts["src_rhetorical_device"].fillna("NA").str.split("|").str[0]
    rh = ts.groupby("rhet")["uplift"].agg(["size", "median"]).query("size >= 150").reset_index()
    out["rhetoric"] = [{"label": r.rhet, "n": int(r["size"]), "median_uplift": _r(r["median"])} for _, r in rh.iterrows() if r.rhet not in ("Other", "NA")]
    se = ts.groupby("src_sentiment_label")["uplift"].agg(["size", "median"]).reset_index()
    out["sentiment"] = [{"label": r.src_sentiment_label, "n": int(r["size"]), "median_uplift": _r(r["median"])} for _, r in se.iterrows()]

    # regression
    reg = pd.concat([fit(df, "Truth Social"), fit(df, "Twitter", min_year="2015-06-16")])
    reg[["group", "label"]] = reg["term"].apply(lambda t: pd.Series(_label(t)))
    out["regression"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in reg.to_dict("records")]

    # flags
    flags = []
    for plat in ["Twitter", "Truth Social"]:
        flags += make_flags(df, "format", "format", plat)
        flags += feature_flags(df, {"has_nickname": "a nickname for an opponent", "heavy_caps": "heavy CAPITALS",
                                    "has_exclamation": "an exclamation mark", "has_link": "a link"}, plat)
    out["flags"] = flags
    return out


_FAN_HANDLE = re.compile(r'^("?)@\w+:')
_TS_USER = re.compile(r'(truthsocial\.com/)(?:users/|@)[A-Za-z0-9_]+')


def mask_private_handles(text: str) -> str:
    """Hide ordinary users' handles before publishing: the fan he quoted in an '@fan: ...' tweet,
    and the account in a shared Truth Social post link. @mentions inside his own words are kept."""
    text = _FAN_HANDLE.sub(r"\1@[fan account]:", text)
    return _TS_USER.sub(r"\1users/[user]", text)


def explorer(df: pd.DataFrame, per_era_top: int = 120, per_era_bottom: int = 40, per_cell: int = 6) -> list[dict]:
    """Curated posts for the explorer: best and worst by uplift in each era, plus the best few
    for every era x topic and era x format cell. Text truncated to 240 characters."""
    d = df[df["uplift"].notna() & df["has_text"] & ~((df["platform"] == "Twitter") & (df["created_utc"] < ANALYSIS_START_TWITTER))].copy()
    d["era_s"] = era_short(d)
    keep = []
    for _, g in d.groupby("era_s"):
        keep.append(g.nlargest(per_era_top, "uplift"))
        keep.append(g.nsmallest(per_era_bottom, "uplift"))
    for _, g in d.groupby(["era_s", "topic"]):
        keep.append(g.nlargest(per_cell, "uplift"))
    for _, g in d.groupby(["era_s", "format"]):
        keep.append(g.nlargest(per_cell, "uplift"))
    k = pd.concat(keep).drop_duplicates("row_id")
    return [{
        "d": r.created_et.strftime("%Y-%m-%d"), "p": "TS" if r.platform == "Truth Social" else "TW", "e": r.era_s,
        "t": mask_private_handles((r.text[:240] + "…") if len(r.text) > 240 else r.text), "topic": r.topic, "fmt": r.format,
        "eng": int(r.eng), "usual": int(r.usual), "x": round(float(r.uplift), 2),
    } for r in k.sort_values("uplift", ascending=False).itertuples()]


def write(df: pd.DataFrame, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(build(df), f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(out_dir, "explorer.json"), "w") as f:
        json.dump(explorer(df), f, ensure_ascii=False, separators=(",", ":"))
