import os
import yaml
from ultralytics import YOLO


"""
YOLO Animal Detection Training Code
Dataset: 7850 train | 951 val | 1282 test images
Classes: Cat, Dog, Bird, Horse, Sheep, Elephant, Bear, Zebra, Giraffe
"""

def tune_yolo():
    """
    Finds the best hyperparameters (learning rate, momentum, weight decay, etc.)
    by testing 100 different combinations. Each combination trains for 30 epochs.
    """
    try:
        # Load pretrained YOLOv11m model
        model = YOLO('yolo11m.pt')

        # Path to your dataset configuration
        yaml_path = './open-images-animals-train/yolo/dataset.yaml'

        # Run hyperparameter tuning
        results = model.tune(
            data=yaml_path,
            iterations=100,  # test 100 different hyperparameter combinations
            epochs=30,  # train each combination for 30 epochs
            imgsz=640,
            batch=24,
            device=0,  # use GPU 0
            val=True,  # validate during tuning
            workers=8,
            plots=True,  # generate training plots
            save=True,  # save checkpoints
            name='yolo11m_tune'  # Output folder name
        )

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def train_yolo(use_tuned_params=False):
    """
    Trains the final model over 100 epochs using either default or tuned hyperparameters.
    """

    try:
        model = YOLO('yolo11m.pt')

        # Path to your dataset configuration
        yaml_path = './open-images-animals-train/yolo/dataset.yaml'

        # Base training configuration
        train_config = {
            'data': yaml_path,
            'epochs': 100,
            'imgsz': 640,
            'batch': 24,
            'device': 0,  # use GPU
            'save': True,  # save checkpoints
            'plots': True,  # generate training plots
            'val': True,  # validate during training
            'workers': 8,
            'amp': True,
            'name': 'yolo11l_finetune'  # output folder name
        }

        # using tuned parameters
        if use_tuned_params:
            tuned_params_path = 'runs/detect/yolo11m_tune/best_hyperparameters.yaml'

            if os.path.exists(tuned_params_path):
                with open(tuned_params_path, 'r') as f:
                    tuned_params = yaml.safe_load(f)
                    # update config with tuned parameters
                    train_config.update(tuned_params)
            else:
                print(f"could not find {tuned_params_path}")

        # train the model
        results = model.train(**train_config)
        return results

    except Exception as e:
        print(f"ERROR: {e}")
        return None

if __name__ == "__main__":
    # uncomment desired functions

    # tune_yolo()
    train_yolo(use_tuned_params=True)
    # train_yolo(use_tuned_params=False)

    # run if crash occurred during final training
    # model=YOLO('./runs/detect/yolo11m_finetune/weights/last.pt')
    # model.train(resume=True)
