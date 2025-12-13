import os
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "./runs/detect/yolo11m_finetune/weights/best.pt"
AUGMENTED_ROOT = "./augmented_testsets"
OUTPUT_DIR = "runs/detect/robustness_eval"

def run_evaluation():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    datasets_to_test = []
    if not os.path.exists(AUGMENTED_ROOT):
        print(f"Error: '{AUGMENTED_ROOT}' not found.")
        return

    subfolders = sorted(os.listdir(AUGMENTED_ROOT))
    for folder in subfolders:
        yaml_path = os.path.join(AUGMENTED_ROOT, folder, "dataset.yaml")
        if os.path.exists(yaml_path):
            datasets_to_test.append((folder, yaml_path))

    if not datasets_to_test:
        print("No datasets found!")
        return

    summary_data = []
    per_class_data = []

    for name, yaml_file in datasets_to_test:
        # run validation
        metrics = model.val(
            data=yaml_file,
            split='val',
            imgsz=640,
            device=0, # use GPU
            plots=True,
            project=OUTPUT_DIR,
            name=name,
            exist_ok=True
        )

        summary_data.append({
            "Dataset": name,
            "mAP@50": round(metrics.box.map50, 4),
            "mAP@50-95": round(metrics.box.map, 4),
            "Precision": round(metrics.box.mp, 4),  # Mean Precision
            "Recall": round(metrics.box.mr, 4),  # Mean Recall
        })

        for class_idx, class_map in enumerate(metrics.box.maps):
            class_name = metrics.names[class_idx]
            per_class_data.append({
                "Dataset": name,
                "Class": class_name,
                "mAP@50-95": round(class_map, 4)
            })

    # save results
    df_summary = pd.DataFrame(summary_data).sort_values(by="Dataset")
    df_summary.to_csv("robustness_summary.csv", index=False)

    df_per_class = pd.DataFrame(per_class_data)
    df_pivot = df_per_class.pivot(index="Dataset", columns="Class", values="mAP@50-95")
    df_per_class.to_csv("robustness_per_class.csv", index=False)

if __name__ == "__main__":
    run_evaluation()