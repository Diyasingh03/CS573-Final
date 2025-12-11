import os, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# -----------------------------
# Utilities
# -----------------------------
def read_category_names(coco_json_path):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    cats = sorted(coco.get("categories", []), key=lambda x: x["id"])
    return [c["name"] for c in cats]

def build_cfg(config_file, weights_path, num_classes, out_dir, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def run_eval(cfg, dataset_name, out_dir):
    model = DefaultTrainer.build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = COCOEvaluator(dataset_name=dataset_name, tasks=["bbox"], distributed=False, output_dir=out_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, val_loader, evaluator)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    return results

def iou_matrix(boxes1, boxes2):
    """
    boxes: (N,4)/(M,4) in XYXY
    return: IoU (N,M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)

    area1 = (b1[:,2]-b1[:,0]).clip(min=0) * (b1[:,3]-b1[:,1]).clip(min=0)
    area2 = (b2[:,2]-b2[:,0]).clip(min=0) * (b2[:,3]-b2[:,1]).clip(min=0)

    inter_x1 = np.maximum(b1[:,None,0], b2[None,:,0])
    inter_y1 = np.maximum(b1[:,None,1], b2[None,:,1])
    inter_x2 = np.minimum(b1[:,None,2], b2[None,:,2])
    inter_y2 = np.minimum(b1[:,None,3], b2[None,:,3])
    inter_w = (inter_x2 - inter_x1).clip(min=0)
    inter_h = (inter_y2 - inter_y1).clip(min=0)
    inter = inter_w * inter_h
    union = area1[:,None] + area2[None,:] - inter + 1e-6
    return inter / union

def reorder_confmat(cm, labels, desired_order):
    order = [c for c in desired_order if c in labels] + [c for c in labels if c not in desired_order]
    idx = [labels.index(c) for c in order]
    cm_re = cm[np.ix_(idx, idx)]
    labels_re = order
    return cm_re, labels_re

def to_axes(cm, x_axis="pred"):  
    if x_axis == "true":
        return cm.T
    return cm


def draw_confmat(cm, labels,save_path, cmap="Blues", annotate_zeros=True):
    fig, ax = plt.subplots(figsize=(1.0 + 0.6*len(labels), 1.0 + 0.6*len(labels)))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix")

    thresh = (np.nanmax(cm) + np.nanmin(cm)) / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0 and not annotate_zeros:
                continue
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def compute_detection_confusion_matrix(cfg, dataset_name, class_names, out_dir,
                                       iou_thresh=0.5, cm_score_thresh=0.0):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    id_to_gt = {}
    for d in dataset_dicts:
        img_id = d.get("image_id", None)
        if img_id is None:
            img_id = d["file_name"]
        gboxes, gclasses = [], []
        for ann in d.get("annotations", []):
            if "bbox" not in ann or "category_id" not in ann:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue
            gboxes.append(_xywh_to_xyxy(ann["bbox"]))
            gclasses.append(int(ann["category_id"]))
        id_to_gt[img_id] = (np.array(gboxes, dtype=np.float32) if gboxes else np.zeros((0,4), dtype=np.float32),
                            np.array(gclasses, dtype=np.int64) if gclasses else np.zeros((0,), dtype=np.int64))

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    loader = build_detection_test_loader(cfg, dataset_name)

    K = len(class_names)
    labels = class_names + ["BG"]
    cm = np.zeros((K+1, K+1), dtype=np.int64)
    per_cls_gt  = np.zeros(K, dtype=np.int64)
    per_cls_pred= np.zeros(K, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            out = model(batch)[0]["instances"].to("cpu")

            img_key = batch[0].get("image_id", None)
            if img_key is None:
                img_key = batch[0]["file_name"]

            gt_boxes, gt_classes = id_to_gt.get(img_key, (np.zeros((0,4),dtype=np.float32),
                                                          np.zeros((0,),dtype=np.int64)))
            if len(out) > 0:
                keep = np.where(out.scores.numpy() >= float(cm_score_thresh))[0]
                pred_boxes = out.pred_boxes.tensor.numpy()[keep]
                pred_classes = out.pred_classes.numpy()[keep]
            else:
                pred_boxes = np.zeros((0,4), dtype=np.float32)
                pred_classes = np.array([], dtype=np.int64)

            for c in gt_classes: per_cls_gt[c] += 1
            for c in pred_classes: per_cls_pred[c] += 1

            iou = iou_matrix(gt_boxes, pred_boxes)
            matched_gt, matched_pred = set(), set()
            pairs = []
            if iou.size > 0:
                g_idx, p_idx = np.where(iou >= float(iou_thresh))
                cand = list(zip(g_idx.tolist(), p_idx.tolist(), iou[g_idx, p_idx].tolist()))
                cand.sort(key=lambda x: x[2], reverse=True)
                for g, p, _ in cand:
                    if g not in matched_gt and p not in matched_pred:
                        matched_gt.add(g); matched_pred.add(p)
                        pairs.append((g, p))


            for g, p in pairs:
                cm[int(gt_classes[g]), int(pred_classes[p])] += 1

            for g in range(len(gt_boxes)):
                if g not in matched_gt:
                    cm[int(gt_classes[g]), K] += 1

            for p in range(len(pred_boxes)):
                if p not in matched_pred:
                    cm[K, int(pred_classes[p])] += 1

    # 輸出
    ds_tag = dataset_name.replace("/", "_")
    cm_csv = os.path.join(out_dir, f"confmat_{ds_tag}.csv")
    cm_png = os.path.join(out_dir, f"confmat_{ds_tag}.png")

    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(cm_csv, index=True)

    desired = ["Cat","Dog","Bird","Horse","Sheep","Elephant","Bear","Zebra","Giraffe","BG"]  
    cm_re, labels_re = reorder_confmat(cm, labels, desired)
    cm_plot = to_axes(cm_re, x_axis="true")
    draw_confmat(cm_plot, labels_re, save_path = cm_png, annotate_zeros=False)

    print(f"Saved confusion matrix csv to: {cm_csv}")

    with open(os.path.join(out_dir, f"confmat_counts_{ds_tag}.json"), "w") as f:
        json.dump({
            "classes": class_names,
            "gt_counts": {class_names[i]: int(per_cls_gt[i]) for i in range(K)},
            "pred_counts": {class_names[i]: int(per_cls_pred[i]) for i in range(K)},
            "bg_row_col": "Last index is BG"
        }, f, indent=2)

    return cm, labels

def visualize_single_image_by_name(
    cfg,
    dataset_name,
    filename,          
    out_dir,
    class_filter=None, 
    score_thresh=0.5
):
    metadata = MetadataCatalog.get(dataset_name)
    thing_classes = metadata.thing_classes
    name2id = {name: i for i, name in enumerate(thing_classes)}

    if class_filter is not None:
        keep_ids = [name2id[c] for c in class_filter if c in name2id]
    else:
        keep_ids = None

    target_basename = os.path.basename(filename)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    record = None
    for d in dataset_dicts:
        if os.path.basename(d["file_name"]) == target_basename:
            record = d
            break

    if record is None:
        print(f"[WARN] image {target_basename} not found in dataset {dataset_name}")
        return

    img_path = record["file_name"]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot read image: {img_path}")
        return

    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        print("No predictions on this image.")
        return

    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    keep = scores >= float(score_thresh)
    if keep_ids is not None:
        keep = keep & np.isin(classes, keep_ids)

    if not np.any(keep):
        print("No predictions pass score/class filter.")
        return

    instances = instances[keep]

    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE
    )
    out = v.draw_instance_predictions(instances)

    save_dir = os.path.join(out_dir, "vis_single")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_{target_basename}")
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print(f"Saved single-image visualization to: {save_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model_final.pth")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--test-img", required=True, help="Directory of test images")
    parser.add_argument("--test-ann", required=True, help="Path to COCO JSON annotations")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--name", default=None, help="Dataset name to register")
    parser.add_argument("--config", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help="Base config used for training")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold at test time")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU for confusion matrix matching")
    parser.add_argument("--cm-score-thresh", type=float, default=0.0, help="Min score to include a prediction into confusion matrix")
    args = parser.parse_args()

    test_img = os.path.abspath(args.test_img)
    test_ann = os.path.abspath(args.test_ann)
    out_dir  = os.path.abspath(args.out)

    assert os.path.isdir(test_img), f"test image dir not found: {test_img}"
    assert os.path.isfile(test_ann), f"annotation json not found: {test_ann}"
    os.makedirs(out_dir, exist_ok=True)

    thing_classes = read_category_names(test_ann)

    ds_name = args.name or f"test_{Path(test_img).name}"
    if ds_name in MetadataCatalog.list():
        from detectron2.data.catalog import DatasetCatalog as DC
        DC.remove(ds_name)
        MetadataCatalog.remove(ds_name)

    register_coco_instances(ds_name, {}, test_ann, test_img)
    MetadataCatalog.get(ds_name).thing_classes = thing_classes

    cfg = build_cfg(
        config_file=args.config,
        weights_path=os.path.abspath(args.weights),
        num_classes=args.num_classes,
        out_dir=out_dir,
        score_thresh=args.score_thresh,
    )
    cfg.DATASETS.TEST = (ds_name,)

    print("\n===== Dataset & Config =====")
    print(f"Dataset name : {ds_name}")
    print(f"Images dir   : {test_img}")
    print(f"Ann json     : {test_ann}")
    print(f"Classes      : {thing_classes}")
    print(f"Weights      : {cfg.MODEL.WEIGHTS}")
    print(f"Config file  : {args.config}")
    print(f"Output dir   : {cfg.OUTPUT_DIR}")
    print("============================\n")

    _ = run_eval(cfg, ds_name, cfg.OUTPUT_DIR)

    print("\nBuilding confusion matrix ...")
    _cm, _labels = compute_detection_confusion_matrix(
        cfg, ds_name, thing_classes, cfg.OUTPUT_DIR,
        iou_thresh=args.iou_thresh, cm_score_thresh=args.cm_score_thresh
    )
    print("Done.")

    visualize_single_image_by_name(
        cfg,
        ds_name,
        filename="0817eb54181b4e01.jpg",      
        out_dir=cfg.OUTPUT_DIR,
        class_filter=None,    
        score_thresh=0.3
    )

if __name__ == "__main__":
    main()
