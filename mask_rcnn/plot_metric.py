import json, os, matplotlib.pyplot as plt

OUT_DIR = "/home/cycsherry/projects/cs573/outputs/maskrcnn_bbox_ft4" 
mj = os.path.join(OUT_DIR, "metrics.json")
ef = os.path.join(OUT_DIR, "eval_final.json")

iters, loss, lr = [], [], []
ap_iters, ap, aps, apm, apl = [], [], [], [], []

if os.path.exists(mj):
    with open(mj) as f:
        for line in f:
            rec = json.loads(line)
            it = rec.get("iteration")
            if it is not None and "total_loss" in rec:
                iters.append(it); loss.append(rec["total_loss"]); lr.append(rec.get("lr"))
            if "bbox/AP" in rec:
                ap_iters.append(it if it is not None else (ap_iters[-1]+1 if ap_iters else 0))
                ap.append(rec["bbox/AP"])
                aps.append(rec.get("bbox/APs")); apm.append(rec.get("bbox/APm")); apl.append(rec.get("bbox/APl"))

if not ap and os.path.exists(ef):
    with open(ef) as f:
        rec = json.load(f)
    ap_iters.append(max(iters) if iters else 0)
    ap.append(rec["bbox"]["AP"])
    aps.append(rec["bbox"]["APs"])
    apm.append(rec["bbox"]["APm"])
    apl.append(rec["bbox"]["APl"])

def savefig(path): 
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

if loss:
    plt.figure(); plt.plot(iters, loss); plt.xlabel("iter"); plt.ylabel("total_loss"); plt.title("Train Loss")
    savefig(os.path.join(OUT_DIR, "loss.png"))

if ap:
    plt.figure(); 
    plt.plot(ap_iters, ap, label="mAP"); 
    if any(aps): plt.plot(ap_iters, aps, label="APs")
    if any(apm): plt.plot(ap_iters, apm, label="APm")
    if any(apl): plt.plot(ap_iters, apl, label="APl")
    plt.xlabel("iter (eval)"); plt.ylabel("AP"); plt.legend(); plt.title("Validation AP")
    savefig(os.path.join(OUT_DIR, "ap.png"))

if lr and any(lr):
    plt.figure(); plt.plot(iters[:len(lr)], lr); plt.xlabel("iter"); plt.ylabel("lr"); plt.title("LR")
    savefig(os.path.join(OUT_DIR, "lr.png"))

print("Saved plots to:", OUT_DIR)
