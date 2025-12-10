#!/usr/bin/env python3
import argparse, json, os
ap=argparse.ArgumentParser(); ap.add_argument("--latex_dir",required=True); ap.add_argument("--metrics",required=True); a=ap.parse_args()
with open(a.metrics) as f: m=json.load(f)
os.makedirs(os.path.join(a.latex_dir,"tables"), exist_ok=True)
with open(os.path.join(a.latex_dir,"tables","results_table.tex"),"w") as f:
    f.write(f"""\\begin{{table}}[t]\n\\centering\n\\begin{{tabular}}{{llll}}\n\\toprule\nmAP@0.5 & mAP@0.5:0.95 & Precision & Recall \\ \\midrule\n{m.get('mAP50')} & {m.get('mAP50-95')} & {m.get('precision')} & {m.get('recall')} \\ \n\\bottomrule\n\\end{{tabular}}\n\\caption{{Single-run results.}}\n\\label{{tab:single}}\n\\end{{table}}\n""")
print("[✓] Wrote LaTeX results_table.tex")
