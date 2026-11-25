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
| 3. Score | `score.py` | **Usual engagement** = median likes + reposts of the account's own original posts (no reposts, no replies) on the same platform within 14 days either side. **Uplift** = engagement ÷ usual. 1.0 = a typical post; 2.0 = twice usual. Same baseline idea as the Knox Index. |
| 4. Model | `model.py` | Regression of log uplift on format, topics, style and tone, so each factor is measured holding the others constant. Reports multipliers with 95% confidence intervals. |
| 5. Flags | `flags.py` | Writes insight sentences from the numbers, e.g. *"Posts with a link do 40% worse than posts without on Truth Social (8,592 posts)."* Only fires when n ≥ 100 and the effect is ≥ 15%. |
| 6. Export | `export.py` | Writes `report.json` (aggregates only) and `explorer.json` (about 1,000 curated example posts, text cut to 240 characters). The raw data is never exported. |
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

Because of this, the report leans on **format, style and tone** (measured exactly) more than on topic, and only reports topic findings that also hold when checked against the Truth Social export's own topic labels (a different, unknown method).

Independent labels came from Claude (Anthropic) acting as a second annotator, blind to the rules' output, then spot-checked by hand.

## Limitations

- Engagement comes from each platform's own audience: a proxy for a politician's base, not a survey of it.
- No view counts in either export; Twitter has no reply counts.
- Observational data: the model shows association, not cause.
- Keyword topics are moderate (see above). Next step: a trained classifier (an LLM with a fixed prompt, or TF-IDF + logistic regression on a few hundred hand labels), validated the same way.

## Extending it to a US Knox Index

The scoring needs only a post time, text and engagement counts, so any account on X, TikTok, Instagram or Facebook can be added through a new loader in `ingest.py`. With views available, the same baseline works on views, as the Knox Index does.

Data files are not included in this repository.
