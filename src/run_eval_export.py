#!/usr/bin/env python3
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
import matplotlib.pyplot as plt
import math

# Candidate header names across Ultralytics versions
CANDIDATES = {
    "precision": ["metrics/precision(B)", "metrics/precision", "precision", "P"],
    "recall":    ["metrics/recall(B)",    "metrics/recall",    "recall",    "R"],
    "mAP50":     ["metrics/mAP50(B)",     "metrics/mAP50",     "mAP50",     "map50", "val/mAP50"],
    "mAP50-95":  ["metrics/mAP50-95(B)",  "metrics/mAP50-95",  "mAP50-95",  "map50-95", "map_0.5:0.95", "val/mAP50-95"],
}

def to_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): 
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None

def pick(last_row: dict, keys: list):
    for k in keys:
        if k in last_row:
            v = to_float(last_row[k])
            if v is not None:
                return v
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to Ultralytics run dir (contains results.csv)")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.run, "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # **Normalize headers**: trim leading/trailing whitespace
    df.rename(columns=lambda c: str(c).strip(), inplace=True)

    if len(df) == 0:
        raise RuntimeError("results.csv is empty")

    last = df.tail(1).to_dict("records")[0]
    summary = {
        "precision": pick(last, CANDIDATES["precision"]),
        "recall":    pick(last, CANDIDATES["recall"]),
        "mAP50":     pick(last, CANDIDATES["mAP50"]),
        "mAP50-95":  pick(last, CANDIDATES["mAP50-95"]),
        "columns_seen": list(df.columns),
    }

    with open(os.path.join(args.out, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Simple PR trace over epochs if both series exist
    x = next((df[k] for k in CANDIDATES["recall"]    if k in df.columns), None)
    y = next((df[k] for k in CANDIDATES["precision"] if k in df.columns), None)

    plt.figure()
    if x is not None and y is not None:
        plt.plot(x, y, linewidth=2)
    else:
        plt.plot([0, 1], [1, 0], "--")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (training trace)"); plt.grid(True)
    plt.savefig(os.path.join(args.out, "pr_curve.png"), dpi=200, bbox_inches="tight")
    print("[✓] Wrote", args.out, summary)

if __name__ == "__main__":
    main()

'''
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(a.run, "results.csv")
    df = pd.read_csv(csv)
    last = df.iloc[-1].to_dict()

    def pick(keys):
        for k in keys:
            if k in last and last[k] is not None:
                v = last[k]
                if isinstance(v, float) and math.isnan(v):
                    continue
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    # Prefer (B) metrics if present, otherwise fallback
    summary = {
        "mAP50": pick(["metrics/mAP50(B)", "metrics/mAP50"]),
        "mAP50-95": pick(["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        "precision": pick(["metrics/precision(B)", "metrics/precision"]),
        "recall": pick(["metrics/recall(B)", "metrics/recall"]),
    }

    with open(os.path.join(a.out, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def find_col(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    rec_col = find_col(["metrics/recall(B)", "metrics/recall"])
    prec_col = find_col(["metrics/precision(B)", "metrics/precision"])

    plt.figure()
    if rec_col and prec_col:
        plt.plot(df[rec_col], df[prec_col], linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR (training trace)")
    plt.grid(True)
    plt.savefig(os.path.join(a.out, "pr_curve.png"), dpi=200, bbox_inches="tight")

    print("Wrote", a.out, summary)


if __name__ == "__main__":
    main()

'''