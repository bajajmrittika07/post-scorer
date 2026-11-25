# post-scorer

**A prototype single-account "US Knox Index": scores every post a politician makes against their own usual performance, tags what the post is and how it is written, and writes plain-English insight flags.**

Built by Mrittika Bajaj for a case study on how US politicians use social media (prepared for the Atlantic Data Institute). First run on 89,325 posts from Donald Trump's Twitter (2009–2021) and Truth Social (2022–2026) accounts.

## Why it exists

Trump's audience grew from a few thousand followers in 2009 to tens of millions. Raw likes mostly tell you *when* a post was made. To learn *what works*, each post has to be compared with what the same account normally got at that time. That is the question this tool answers for every post: did it beat the account's usual?

## What it does

| Step | Module | What happens |
|---|---|---|
| 1. Ingest | `ingest.py` | Loads both exports into one schema (platform, time in UTC and US Eastern, text, likes, reposts, replies). Rebuilds the repost flag from the text (the Truth Social export's own flag is False for every row). Repairs broken text encoding. Classifies format from the link domain (text only, article link, video link, shared post, media with no caption, repost). |
| 2. Tag | `tag.py` | **Topic**: 14 transparent keyword rules (multi-label, plus one display label by priority). **Style**: share of words in CAPITALS, exclamation marks, questions, opponent nicknames, calls to action, sign-offs, length. **Timing**: hour and weekday in US Eastern time. |
| 3. Score | `score.py` | **Usual engagement** = median likes + reposts of the account's own original posts (no reposts, no replies) on the same platform within 14 days either side. **Uplift** = engagement ÷ usual. 1.0 = a typical post; 2.0 = twice usual. Same baseline idea as the Knox Index. **Exclusions** (broken counts, not low engagement): deleted tweets, whose engagement froze at deletion; tweets Twitter flagged in 2020, on some of which it switched off likes and retweets; and any post with engagement under 2% of usual. Deleted and flagged tweets are also left out of the baseline. Repeated texts are kept. |
| 4. Model | `model.py` | Regression of log uplift on format, topics, style, tone and how many posts he made that day, so each factor is measured holding the others constant. Each doubling of posts that day is linked to 15% less engagement per post on Truth Social and 4% less on Twitter. Reports multipliers with 95% confidence intervals. |
| 5. Flags | `flags.py` | Writes insight sentences from the numbers, e.g. *"Posts with a link do 40% worse than posts without on Truth Social (8,586 posts)."* Only fires when n ≥ 100 and the effect is ≥ 15%. |
| 6. Export | `export.py` | Writes `report.json` (aggregates only) and `explorer.json` (805 curated example posts, text cut to 240 characters; handles of ordinary users masked). The raw data is never exported. |
| Validate | `validate.py` | Scores the topic rules against any hand-labelled CSV (agreement, Cohen's kappa, per-topic report). |

## Run it

```bash
pip install -r requirements.txt
python -m post_scorer.cli --twitter trump_tweets.csv --truth trump_posts_full.csv --out ../report/data
python -m post_scorer.validate hand_check.csv --truth-col your_label
```

Runtime about one minute on a laptop.

## How well does the topic tagger work?

Tested honestly, against posts it was not built from:

| Round | Sample | Exact agreement with independent labels | Cohen's kappa |
|---|---|---|---|
| v1 rules | 150 posts (30 per era) | 49% | 0.42 |
| v2 rules, same sample (after fixing errors) | 150 | 70% (overfit, not a fair score) | — |
| **v2 rules, fresh held-out sample** | **150 new posts** | **54%** (Twitter 59%, Truth Social 47%) | **0.49** |

The correct topic is among the topics the rules flag 66% of the time. Hardest cases: bare news links and quoted headlines on Truth Social.

Independent labels came from Claude (Anthropic) acting as a second annotator, blind to the rules' output.

**Blind hand check, round 1.** Mrittika labelled 89 of the held-out posts herself, without seeing either label: all 69 where the rules and Claude disagreed, plus 20 where they agreed. Weighted back to the full 150, the rules match her labels about 51% of the time and Claude about 58% (estimates). The rules' main weakness is coverage: in 22 of their 58 misses, no keyword matched at all.

**Blind hand check, round 2** (`hand_check.csv`): 100 new random scored posts, 50 per platform, labelled by Mrittika without seeing either label. Because the sample is random, no weighting is needed.

| | Rules | Claude |
|---|---|---|
| Agreement with hand labels | **50%** (kappa 0.45) | **80%** (kappa 0.78) |
| Truth Social / Twitter | 54% / 46% | 76% / 84% |

The hand-labelled topic is among the topics the rules flag 66% of the time. In 16 of the rules' 50 misses no keyword matched at all. Per topic: **Endorsements** rule precision 4 of 5 posts, recall 4 of 5. **Patriotism, faith & holidays**: the rules gave no post this label and only 2 posts were hand-labelled with it, too few to estimate precision. Claude's most common miss was calling thanks and self-promotion posts "Media" (5).

**Cross-check against the Truth Social export's own topic labels.** The export carries its own `topic` field, made by an undocumented method. Mapping our topics to its labels where an equivalent exists, the two agree on **68%** of 11,860 scored text posts:

| Our topic | Counts as a match if the export says | Agreement |
|---|---|---|
| Legal cases & investigations | Legal | 82% |
| Endorsements | Elections | 78% |
| Elections & campaign | Elections | 67% |
| Immigration & border | Immigration | 64% |
| Foreign policy & military | Foreign Policy, Military or National Security | 63% |
| Economy, trade & prices | Economy | 62% |
| Media | Media | 57% |
| Crime & law and order | Crime or Legal | 37% |
| Patriotism, faith & holidays | Religion | 33% |

Opponents, Sport & culture, Business and Thanks have no equivalent in the export and are left out; Covid (20 posts) has none in practice. Because topic tagging is moderate, the report leans on **format, style and tone**, which are measured exactly, and treats topic findings as a second line of evidence.

## Limitations

- Engagement comes from each platform's own audience: a proxy for a politician's base, not a survey of it.
- No view counts in either export; Twitter has no reply counts.
- Observational data: the model shows association, not cause.
- Keyword topics are moderate (see above). Next step: a trained classifier (an LLM with a fixed prompt, or TF-IDF + logistic regression on a few hundred hand labels), validated the same way.

## Extending it to a US Knox Index

The scoring needs only a post time, text and engagement counts, so any account on X, TikTok, Instagram or Facebook can be added through a new loader in `ingest.py`. With views available, the same baseline works on views, as the Knox Index does.

Data files are not included in this repository.
