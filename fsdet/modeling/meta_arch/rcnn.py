# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from fsdet.structures import ImageList
from fsdet.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from fsdet.utils.events import get_event_storage
from fsdet.utils.visualizer import Visualizer
from fsdet.data.detection_utils import convert_image_to_rgb
from fsdet.modeling.utils import concat_all_gathered

from fsdet.utils.GEM import GeM

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

        # s1 = 0
        # s2 = 0
        # for model in [self.backbone.fpn_lateral2, self.backbone.fpn_lateral3, self.backbone.fpn_lateral4, self.backbone.fpn_lateral5]:
        #     s1 += sum(p.numel() for p in model.parameters())
        # for model in [self.backbone.fpn_output2, self.backbone.fpn_output3, self.backbone.fpn_output4, self.backbone.fpn_output5]:
        #     s2 += sum(p.numel() for p in model.parameters())
        # print('FPN',s1, s2)
        if cfg.INPUT.USE_MIXUP:
            self.use_mixup = True
        else:
            self.use_mixup = False

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

        # self.GeM2 = GeM()
        # self.GeM3 = GeM()
        # self.GeM4 = GeM()
        # self.GeM5 = GeM()
        # self.GeM6 = GeM()


        # self.module1 = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         nn.BatchNorm2d(64, momentum=1, affine=True),
        #         nn.ReLU(),
        #         GeM(),
        #         # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         # GeM(),
        #     )
        # self.module1_score = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Sigmoid(),
        # )
        #
        # self.module2 = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         nn.BatchNorm2d(64, momentum=1, affine=True),
        #         nn.ReLU(),
        #         GeM(),
        #         # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         # GeM(),
        #     )
        # self.module2_score = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Sigmoid(),
        # )
        #
        # self.module3 = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         nn.BatchNorm2d(64, momentum=1, affine=True),
        #         nn.ReLU(),
        #         GeM(),
        #         # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         # GeM(),
        #     )
        # self.module3_score = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Sigmoid(),
        # )
        #
        # self.module4 = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         nn.BatchNorm2d(64, momentum=1, affine=True),
        #         nn.ReLU(),
        #         GeM(),
        #         # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         # GeM(),
        #     )
        # self.module4_score = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Sigmoid(),
        # )
        #
        # self.module5 = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         nn.BatchNorm2d(64, momentum=1, affine=True),
        #         nn.ReLU(),
        #         GeM(),
        #         # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=3, bias=False),
        #         # GeM(),
        #     )
        # self.module5_score = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Sigmoid(),
        # )
        #
        # self.module1.cuda()
        # self.module1_score.cuda()
        # self.module2.cuda()
        # self.module2_score.cuda()
        # self.module3.cuda()
        # self.module3_score.cuda()
        # self.module4.cuda()
        # self.module4_score.cuda()
        # self.module5.cuda()
        # self.module5_score.cuda()




    def forward(self, batched_inputs, batched_inputs_aux=None, lambda_=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        if batched_inputs_aux is not None:
            with torch.no_grad():
                images_aux = self.preprocess_image(batched_inputs_aux)
                # features_aux = self.backbone(images_aux.tensor)
        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor, images_aux.tensor)  # List of L, FPN features

        # if batched_inputs_aux is not None:
        #     with torch.no_grad():
        #         images_aux = self.preprocess_image(batched_inputs_aux)
        #         features_aux = self.backbone(images_aux.tensor)

        'Relation score'
        # score_pooling_p2 = self.module1(features_aux["p2"])
        # score_pooling_p2 = score_pooling_p2.squeeze()
        # score_p2 = self.module1_score(score_pooling_p2)
        # score_p2 = score_p2.unsqueeze(dim=2).unsqueeze(dim=3)
        # # features["p2"] = score_p2 * features["p2"]
        # features["p2"] = score_p2*features_aux["p2"] + (1-score_p2)*features["p2"]
        #
        # score_pooling_p3 = self.module2(features_aux["p3"])
        # score_pooling_p3 = score_pooling_p3.squeeze()
        # score_p3 = self.module2_score(score_pooling_p3)
        # score_p3 = score_p3.unsqueeze(dim=2).unsqueeze(dim=3)
        # # features["p3"] = score_p3 * features["p3"]
        # features["p3"] = score_p3*features_aux["p3"] + (1-score_p3)*features["p3"]
        #
        # score_pooling_p4 = self.module3(features_aux["p4"])
        # score_pooling_p4 = score_pooling_p4.squeeze()
        # score_p4 = self.module3_score(score_pooling_p4)
        # score_p4 = score_p4.unsqueeze(dim=2).unsqueeze(dim=3)
        # # features["p4"] = score_p4 * features["p4"]
        # features["p4"] = score_p4*features_aux["p4"] + (1-score_p4)*features["p4"]
        #
        # score_pooling_p5 = self.module4(features_aux["p5"])
        # score_pooling_p5 = score_pooling_p5.squeeze()
        # score_p5 = self.module4_score(score_pooling_p5)
        # score_p5 = score_p5.unsqueeze(dim=2).unsqueeze(dim=3)
        # # features["p5"] = score_p5 * features["p5"]
        # features["p5"] = score_p5*features_aux["p5"] + (1-score_p5)*features["p5"]
        #
        # score_pooling_p6 = self.module5(features_aux["p6"])
        # score_pooling_p6 = score_pooling_p6.squeeze()
        # score_p6 = self.module5_score(score_pooling_p6)
        # score_p6 = score_p6.unsqueeze(dim=2).unsqueeze(dim=3)
        # # features["p6"] = score_p6 * features["p6"]
        # features["p6"] = score_p6*features_aux["p6"] + (1-score_p6)*features["p6"]


        'GeM'
        # features["p2"] = self.GeM2(features_aux["p2"]) * features["p2"]
        # features["p3"] = self.GeM3(features_aux["p3"]) * features["p3"]
        # features["p4"] = self.GeM4(features_aux["p4"]) * features["p4"]
        # features["p5"] = self.GeM5(features_aux["p5"]) * features["p5"]
        # features["p6"] = self.GeM6(features_aux["p6"]) * features["p6"]


        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, self.use_mixup, lambda_)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 40

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class Yolo(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.INPUT.VIS_PERIOD
        self.device = cfg.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)



@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # resize the raw outputs of an R-CNN detector to produce outputs according to the desired output resolution.
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
