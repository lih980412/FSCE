import multiprocessing as mp

import torch

from fsdet.config import get_cfg
from fsdet import model_zoo
from fsdet.engine import DefaultPredictor
from fsdet.utils.visualizer import Visualizer
from fsdet.data import MetadataCatalog
from fsdet.utils.visualizer import ColorMode

import numpy as np
from PIL import Image
import cv2

class mask_rcnn(object):
    def __init__(self, **kwargs):
        self.cfg = self.setup()
        self.predictor = self.generate(self.cfg)

    def generate(self, cfg):
        # model_path = 'E:/pytorchmr/output-20210910T123916Z-001/output/model_final.pth'
        # model_path = r'D:\UserD\Li\FSCE-1\checkpoints\DiBei_dla60\22_06_03_FSCE_moli_BiFPN_GN_rpnCIoU\model_final.pth'
        # assert model_path.endswith('.pth')
        predictor = DefaultPredictor(cfg)
        return predictor

    def setup(self):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(r"D:\UserD\Li\FSCE-1\configs\Base-RCNN-FPN.yaml")
        cfg.MODEL.WEIGHTS = r"D:\UserD\Li\FSCE-1\checkpoints\DiBei_dla60\22_06_03_FSCE_moli_BiFPN_GN_rpnCIoU\model_final.pth"

        cfg.MODEL.GroupNorm = True

        cfg.MODEL.DLA.ARCH = "DLA-60"
        cfg.MODEL.BACKBONE.NAME = "build_dla_fpn_backbone"

        cfg.MODEL.FPN.IN_FEATURES = ["level2", "level3", "level4", "level5"]

        cfg.MODEL.ROI_HEADS.NAME = "ContrastiveROIHeads"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.OUTPUT_LAYER = "CosineSimOutputLayers"
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.ENABLE = True
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE = 0.2
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT = 0.5
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED = True
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS = [ 6000, 10000 ]
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE = 0.5
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD = 0.8
        cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC = "exp"

        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1., 0.5, 0.25]]

        cfg.DATASETS.TRAIN = ("DiBei_train_only",)
        cfg.DATASETS.TEST = ("DiBei_val_only",)
        cfg.INPUT.RESIZE = True
        cfg.INPUT.RESIZE_VAL = (512, 1024)


        cfg.TEST.DETECTIONS_PER_IMAGE = 10
        # cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = "E:/Study/mask_interface/output/model_final.pth"



        cfg.freeze()

        return cfg

    # def visual_mask(self, image, outputs):
    #     vis = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=0.8, instance_mode = ColorMode.IMAGE)
    #     img_predict = vis.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    #     return img_predict


    def inference(self, image):
        #image应为BGR格式
        outputs = self.predictor(image)
        vis = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=0.8, instance_mode = ColorMode.IMAGE)
        img_predict = vis.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        return img_predict


infere = mask_rcnn()
img = Image.open(r"D:\UserD\Li\FSCE-1\datasets\DiBei\image\22B02283B40_remap.png")
img_predict = infere.inference(image=np.array(img)[:,:, ::-1])
cv2.imshow("1", img_predict)
cv2.waitKey(0)