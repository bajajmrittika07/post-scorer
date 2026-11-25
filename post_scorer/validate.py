"""Score the topic rules against hand labels.

python -m post_scorer.validate data/processed/validation/hand_check.csv --truth-col your_label
Prints exact agreement, Cohen's kappa and a per-topic report. Rows with an empty label are skipped.
"""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--truth-col", default="your_label")
    ap.add_argument("--pred-col", default="rule_label")
    ap.add_argument("--also", default="claude_label", help="second annotator column to compare with, if present")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d = d[d[a.truth_col].fillna("").str.strip() != ""]
    print(f"{len(d)} labelled rows")
    for col in [a.pred_col, a.also]:
        if col in d:
            acc = (d[col] == d[a.truth_col]).mean()
            print(f"{col:>14} vs {a.truth_col}: agreement {acc:.1%}, kappa {cohen_kappa_score(d[col], d[a.truth_col]):.2f}")
    print(classification_report(d[a.truth_col], d[a.pred_col], zero_division=0))


if __name__ == "__main__":
    main()
