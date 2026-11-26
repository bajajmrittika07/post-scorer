"""Command line: python -m post_scorer.cli --twitter PATH --truth PATH --out DIR"""
import argparse
import time

from .ingest import load_all
from .tag import tag
from .score import add_baseline
from .export import write


def main():
    ap = argparse.ArgumentParser(description="Score Trump's posts against his own usual engagement and export report data.")
    ap.add_argument("--twitter", required=True, help="trump_tweets.csv")
    ap.add_argument("--truth", required=True, help="trump_posts_full.csv")
    ap.add_argument("--out", default="report/data", help="folder for report.json and explorer.json")
    ap.add_argument("--peers", default=None, help="optional hand-collected follower CSV (kept private); adds a comparison to report.json")
    ap.add_argument("--save-scored", default=None, help="optional path to save the full scored table (keep private)")
    a = ap.parse_args()
    t0 = time.time()
    df = load_all(a.twitter, a.truth); print(f"ingest: {len(df):,} posts")
    df = tag(df); print("tag: topics, style, timing")
    df = add_baseline(df); print(f"score: {df['uplift'].notna().sum():,} posts scored against their ±14-day baseline")
    if a.save_scored:
        df.to_pickle(a.save_scored)
    write(df, a.out, peers=a.peers); print(f"export: {a.out}/report.json, explorer.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
