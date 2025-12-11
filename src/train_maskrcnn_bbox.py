from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import json, os


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

USER = "cycsherry"  # ubuntu username 
ROOT = f"/home/{USER}/projects/cs573"
TRAIN_IMG = f"{ROOT}/data/images/train"
VAL_IMG   = f"{ROOT}/data/images/val"
TRAIN_JSON= f"{ROOT}/data/images/train/labels.json"
VAL_JSON  = f"{ROOT}/data/images/val/labels.json"
OUT_DIR   = f"{ROOT}/outputs/maskrcnn_bbox"


register_coco_instances("animals_train", {}, TRAIN_JSON, TRAIN_IMG)
register_coco_instances("animals_val",   {}, VAL_JSON,   VAL_IMG)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.MASK_ON = False  
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
cfg.MODEL.WEIGHTS = f"{ROOT}/outputs/maskrcnn_bbox/model_final.pth" ## fine-tuning  

cfg.MODEL.DEVICE = "cuda"  

cfg.DATASETS.TRAIN = ("animals_train",)
cfg.DATASETS.TEST  = ("animals_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 2.5e-4
# cfg.SOLVER.MAX_ITER = 9000
# cfg.SOLVER.STEPS = (6000, 8000)
# cfg.SOLVER.GAMMA = 0.1

# ====== Fine-tuning ======
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 5e-4              
cfg.SOLVER.WARMUP_ITERS = 1000 
cfg.SOLVER.WARMUP_FACTOR = 1.0/1000
cfg.SOLVER.MAX_ITER = 16000

# cfg.SOLVER.STEPS = (8000, 10700)        
cfg.SOLVER.STEPS = (8000, 12000, 14500)   
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 1e-4
# ===============================================

cfg.SOLVER.AMP.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

# checkpoint
cfg.TEST.EVAL_PERIOD = 800 
cfg.SOLVER.CHECKPOINT_PERIOD = 800 

# ===========================

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST  = 640
cfg.INPUT.MAX_SIZE_TEST  = 1024

# cfg.OUTPUT_DIR = OUT_DIR

# Output directionay 
cfg.OUTPUT_DIR = f"{ROOT}/outputs/maskrcnn_bbox_ft4"

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

trainer = Trainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

evaluator = COCOEvaluator("animals_val", cfg, False, output_dir=OUT_DIR)
val_loader = build_detection_test_loader(cfg, "animals_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

#===================
evaluator = COCOEvaluator("animals_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "animals_val")
res = inference_on_dataset(trainer.model, val_loader, evaluator)  # dict
with open(os.path.join(cfg.OUTPUT_DIR, "eval_final.json"), "w") as f:
    json.dump(res, f, indent=2)
print("saved to", os.path.join(cfg.OUTPUT_DIR, "eval_final.json"))
