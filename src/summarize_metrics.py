import json, os, re
import numpy as np

MET = "/home/cycsherry/projects/cs573/outputs/maskrcnn_bbox_ft3/metrics.json"
rows = [json.loads(l) for l in open(MET)]
evals = [r for r in rows if r.get("bbox/AP") is not None]
if not evals:
    print("No eval records in metrics.json"); exit()

iters = [r["iteration"] for r in evals]
ap    = [r["bbox/AP"] for r in evals]
ap50  = [r.get("bbox/AP50") for r in evals]
ap75  = [r.get("bbox/AP75") for r in evals]

best = int(np.argmax(ap))
print(f"Num eval points: {len(ap)}")
print(f"Best iter: {iters[best]}  AP: {ap[best]:.2f}  AP50: {ap50[best]:.2f}  AP75: {ap75[best]:.2f}")
print(f"Last iter: {iters[-1]} AP: {ap[-1]:.2f} AP50: {ap50[-1]:.2f} AP75: {ap75[-1]:.2f}")
