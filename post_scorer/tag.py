"""Tag each post with topic, style and timing features.

Topics come from a transparent keyword dictionary (TOPIC_RULES). A post can match several
topics (topic_* boolean columns); `topic` is the first match in priority order, used for display.
The same rules run on both platforms, so Twitter and Truth Social are labelled the same way.
"""
import re
import pandas as pd

# Order matters for the single display label: more specific topics first.
TOPIC_RULES = {
    "Endorsements": r"\bendorse|complete and total|has my complete|total endorsement|total and complete",
    "Legal cases & investigations": r"witch ?hunt|indict|activist judge|corrupt judge|radical.{0,20}judge|judge (merchan|engoron|chutkan|cannon)|prosecut|\bdoj\b|\bfbi\b|mueller|impeach|jack smith|bragg|fani|willis|letitia|\bjames.s case|\btrial\b|lawfare|special counsel|mar-a-lago raid|\bgag order|e\\. jean|carroll|hoax|russiagate|comey|strzok|brennan|surveillance|epstein|deep state|persecut|weaponiz|ballot.{0,30}(block|remov)|14th amendment|\bindicted|\bappeal",
    "Immigration & border": r"\bborder|\bwall\b|illegal (alien|immigra|migra)|illegals|immigra|migrant|deport|caravan|\bice\b|sanctuary|asylum|invasion|amnesty",
    "Crime & law and order": r"\bcrime|criminal|police|law and order|murder|\bgang|cartel|fentanyl|\bdrugs?\b|shooting|\briot|terroris|safe again",
    "Economy, trade & prices": r"econom|\bjobs\b|jobs (number|report)|tariff|\btrade\b|stock market|\bdow\b|nasdaq|inflation|\bprices?\b|gas price|\btax(es| cut)|unemploy|manufactur|\bgdp\b|interest rate|federal reserve|\bfed\b|powell|grocer|industrial|\bsubsid|\binvest|\bgrant|\boil\b|energy|\bbonds?\b",
    "Foreign policy & military": r"\biran|russia|ukrain|israel|\bgaza|hamas|nato|\bwar\b|military|troops|putin|zelensk|north korea|kim jong|china|chinese|\bxi\b|venezuela|syria|isis|afghan|taliban|libya|missile|nuclear|mexic|canada|carney|trudeau|australia|europe|\beu\b|prime minister|president (of|xi|putin|zelensk|macron|milei|lula|sheinbaum|erdo)(?! the united states)|board of peace|von der leyen|macron|starmer|\bindia\b|japan|tpp",
    "Media": r"fake news|\bmedia\b|\bcnn\b|nytimes|new york times|washington post|msnbc|\bnbc\b|\babc\b|\bcbs\b|reporter|journalist|enemy of the people|failing @?nyt|the view|60 ?minutes|correspondents|fact.?check|newsbusters|censor|\bpress\b|apologi",
    "Opponents": r"biden|obama|hillary|clinton|kamala|harris|pelosi|schumer|newsom|\baoc\b|ocasio|democrat|radical left|\bdems?\b|desantis|haley|romney|mcconnell|\brino|schiff|warren|sanders|bernie|walz|mamdani|cheney|\bbush\b|jeb|karl ?rove|\bthe pres\b",
    "Elections & campaign": r"\bvote|voting|voter|election|ballot|\bpoll|primary|primaries|caucus|campaign|rally|\brallies|nominee|debate|midterm|swing state|\bwin\b|\bwon\b|landslide|rigged|\bcrowd|big speech|speech tonight|join me|see you (soon|tonight)|trump ?2016|trump ?2020|trump ?2024|makeamericagreatagain|\bmaga\b|for president|\brun\b.{0,20}(trump|mr|donald)|please run|great president|republican party|\bgop\b|\brnc\b|super ?pac",
    "Covid": r"covid|coronavirus|\bvirus|vaccine|pandemic|lockdown|\bmasks?\b",
    "Sport & culture": r"stanley ?cup|super ?bowl|champions?\b|\bgame\b|\bnfl\b|\bnba\b|\bufc\b|world series|olympic|\bsoccer|\bboxing|\bpga\b|\bliv\b|\bmasters\b",
    "Patriotism, faith & holidays": r"merry christmas|christmas|happy (new year|thanksgiving|easter|4th|fourth|independence|birthday|father|mother|veterans)|thanksgiving|memorial day|veterans day|\bgod bless|thank god|\bin god\b|\bflag\b|national anthem|\bfaith\b|christian|\bpray|\bchurch",
    "Business, brand & celebrity": r"apprentice|boardroom|ivanka|trump (tower|hotel|international|national|doral|turnberry|winery|university|org|brand)|\bgolf|holes? in one|\bmiss (usa|universe)|\bbook\b|think like a champion|letterman|celebrity|\bhotel|\bresort|\bcondo|realty|real estate|entrepreneur|\bsuccess|\bbusiness(es)?\b",
    "Thanks & self-promotion": r"\bthank|great honor|congratulat|tune in|\bwatch\b|interview|\btonight\b|will be on|truth social|\bratings\b|enjoy!|i joined|\bon @?fox",
}
TOPIC_PATTERNS = {k: re.compile(v, re.I) for k, v in TOPIC_RULES.items()}
RULES_VERSION = 2  # v1 scored 49% vs independent labels; v2 after error analysis (see README)

# Nicknames for opponents (public, widely reported). Used only as a style feature.
NICKNAMES = re.compile(
    r"crooked (hillary|joe)|sleepy joe|\blyin'?\s(ted|kamala)|little marco|pocahontas|crazy (nancy|bernie|joe|kamala)|"
    r"cryin'? chuck|mini mike|shifty schiff|deranged jack|desanctimonious|birdbrain|comrade kamala|tampon tim|"
    r"sloppy steve|meatball ron|newscum|low energy jeb|crooked hillary", re.I)
CALL_TO_ACTION = re.compile(r"\bvote\b|get out and|register|\bdonate|\bsign up|\bjoin\b|tune in|\bwatch\b|go to\b|\bcall your|\bshare\b|\bfollow\b", re.I)
SIGNOFF = re.compile(r"make america great again|\bmaga\b|america first|keep america great", re.I)


def add_topics(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("")
    cols = []
    for name, pat in TOPIC_PATTERNS.items():
        col = "topic_" + re.sub(r"\W+", "_", name.lower()).strip("_")
        df[col] = text.apply(lambda t, p=pat: bool(p.search(t)))
        cols.append((name, col))
    df["topic"] = "Other / unclear"
    df.loc[~df["has_text"], "topic"] = "No text (media only)"
    for name, col in reversed(cols):  # reversed so the highest-priority match is written last
        df.loc[df[col], "topic"] = name
    df["n_topics"] = df[[c for _, c in cols]].sum(axis=1)
    return df


def add_style(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("")
    words = text.str.findall(r"[A-Za-z][A-Za-z'’]+")
    n_words = words.str.len()
    caps_words = words.apply(lambda ws: sum(1 for w in ws if len(w) >= 3 and w.isupper()))
    df["n_words"] = n_words
    df["caps_share"] = (caps_words / n_words.where(n_words > 0)).fillna(0)
    df["heavy_caps"] = df["caps_share"] >= 0.25           # a quarter or more of words in CAPITALS
    df["exclamations"] = text.str.count("!")
    df["has_exclamation"] = df["exclamations"] > 0
    df["has_question"] = text.str.contains(r"\?")
    df["has_nickname"] = text.apply(lambda t: bool(NICKNAMES.search(t)))
    df["has_call_to_action"] = text.str.contains(CALL_TO_ACTION)
    df["has_signoff"] = text.str.contains(SIGNOFF)
    df["length"] = pd.cut(n_words, [-1, 0, 15, 50, 100000], labels=["No text", "Short (1–15 words)", "Medium (16–50)", "Long (51+)"])
    return df


def add_timing(df: pd.DataFrame) -> pd.DataFrame:
    et = df["created_et"]
    df["hour_et"] = et.dt.hour
    df["weekday"] = et.dt.day_name()
    df["late_night"] = df["hour_et"].between(0, 5)       # midnight–6am Eastern
    return df


def tag(df: pd.DataFrame) -> pd.DataFrame:
    return add_timing(add_style(add_topics(df)))
