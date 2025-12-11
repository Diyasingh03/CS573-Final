import os, json, argparse, csv
from collections import defaultdict

def load_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {im["id"]: im.get("file_name", str(im["id"])) for im in data.get("images", [])}
    anns = data.get("annotations", [])
    return cats, images, anns

def count_instances_and_images(cats, images, anns):
    inst_cnt  = defaultdict(int)
    img_sets  = defaultdict(set)  

    for ann in anns:
        cid = ann["category_id"]
        cname = cats.get(cid, f"cat_{cid}")
        inst_cnt[cname] += 1
        img_id = ann["image_id"]
        img_sets[cname].add(img_id)

    image_cnt = {k: len(v) for k, v in img_sets.items()}
    totals = {"images": len(images), "instances": len(anns)}
    for cname in cats.values():
        inst_cnt.setdefault(cname, 0)
        image_cnt.setdefault(cname, 0)
    return inst_cnt, image_cnt, totals

def write_wide_csv(out_csv, category_list, dataset_order, per_ds_counts):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["category"] + dataset_order
        w.writerow(header)
        for cat in category_list:
            row = [cat] + [per_ds_counts[ds].get(cat, 0) for ds in dataset_order]
            w.writerow(row)

def main():
    parser = argparse.ArgumentParser(
        description="Count per-category distributions for multiple COCO datasets."
    )
    parser.add_argument(
        "--set",
        action="append",
        required=True,
        help="Format: NAME=/path/to/annotations.json"
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs/category_counts",
        help="輸出資料夾（預設 ./outputs/category_counts）"
    )
    args = parser.parse_args()

    sets = []
    for s in args.set:
        if "=" not in s:
            raise ValueError(f"--set error：{s}, correct: NAME=/path/to/ann.json")
        name, jpath = s.split("=", 1)
        jpath = os.path.abspath(jpath)
        if not os.path.isfile(jpath):
            raise FileNotFoundError(f"cannot find：{jpath}")
        sets.append((name, jpath))

    union_categories = set()
    ds_order = []
    per_ds_inst = {}   
    per_ds_img  = {}   
    per_ds_tot  = {}   

    print("\n=== Counting per-category distributions ===")
    for name, jpath in sets:
        ds_order.append(name)
        cats, images, anns = load_coco(jpath)
        inst_cnt, image_cnt, totals = count_instances_and_images(cats, images, anns)
        per_ds_inst[name] = dict(inst_cnt)
        per_ds_img[name]  = dict(image_cnt)
        per_ds_tot[name]  = dict(totals)
        union_categories.update(cats.values())

        print(f"- {name}")
        print(f"  ann json   : {jpath}")
        print(f"  #images    : {totals['images']}")
        print(f"  #instances : {totals['instances']}")
    print("==========================================\n")

    category_list = sorted(list(union_categories))

    os.makedirs(args.out_dir, exist_ok=True)
    inst_csv = os.path.join(args.out_dir, "per_category_instances.csv")
    img_csv  = os.path.join(args.out_dir, "per_category_images.csv")

    write_wide_csv(inst_csv, category_list, ds_order, per_ds_inst)
    write_wide_csv(img_csv,  category_list, ds_order, per_ds_img)

    overview_csv = os.path.join(args.out_dir, "dataset_overview.csv")
    with open(overview_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "num_images", "num_instances"])
        for ds in ds_order:
            w.writerow([ds, per_ds_tot[ds]["images"], per_ds_tot[ds]["instances"]])

    print("Saved:")
    print(f"- {inst_csv}  (Instance)")
    print(f"- {img_csv}   (Images)")
    print(f"- {overview_csv} (overview)")
    print("\nDone.")

if __name__ == "__main__":
    main()
