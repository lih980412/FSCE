# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import time
import fsdet.utils.comm as comm
import numpy
import numpy as np
import torch
# from numba import jit
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

from fsdet.utils.LBP import circular_LBP, rotation_invariant_LBP, uniform_pattern_LBP, multi_scale_block_LBP
from skimage.feature import hog, local_binary_pattern
from fsdet.utils.LoG import LoG

from fsdet.utils.GEM import GeM

from skimage.util.dtype import img_as_float
from sklearn.preprocessing import MinMaxScaler

from skimage.feature import _hoghistogram


__all__ = ["GeneralizedRCNN", "ProposalNetwork", "MyRCNN"]


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

        if cfg.MODEL.BACKBONE.FREEZE_FIRST2:
            for p in self.backbone.bottom_up.base_layer:
                p.requires_grad = False
            for p in self.backbone.bottom_up.level0:
                p.requires_grad = False
            for p in self.backbone.bottom_up.level1:
                p.requires_grad = False

            print("froze DLA first 2 module parameters")


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

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        # features = self.backbone(images.tensor, images_aux.tensor)  # List of L, FPN features
        features = self.backbone(images.tensor)  # List of L, FPN features

        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
            if self.use_mixup:
                gt_instances_reverse = [x["instances"].to(self.device) for x in batched_inputs[::-1]]
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
            if self.use_mixup:
                proposals_reverse, proposal_losses_reverse = self.proposal_generator(images, features, gt_instances_reverse)
                proposal_losses['loss_rpn_cls'] = proposal_losses['loss_rpn_cls'] * lambda_ + proposal_losses_reverse['loss_rpn_cls'] * (1-lambda_)
                proposal_losses['loss_rpn_loc'] = proposal_losses['loss_rpn_loc'] * lambda_ + proposal_losses_reverse['loss_rpn_loc'] * (1 - lambda_)

        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances

        if batched_inputs_aux is not None:
            images_aux = self.preprocess_image(batched_inputs_aux)
            featureds_aux = self.backbone(images_aux.tensor)
            gt_instances_aux = [x["instances"].to(self.device) for x in batched_inputs_aux]
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, featureds_aux, gt_instances_aux)
            if self.use_mixup:
                _, detector_losses_reverse = self.roi_heads(images, features, proposals_reverse, gt_instances_reverse, featureds_aux, gt_instances_aux)
                detector_losses['loss_box_reg'] = detector_losses['loss_box_reg'] * lambda_ + detector_losses_reverse['loss_box_reg'] * (1 - lambda_)
                detector_losses['loss_cls'] = detector_losses['loss_cls'] * lambda_ + detector_losses_reverse['loss_cls'] * (1 - lambda_)
                detector_losses['query_loss'] = detector_losses['query_loss'] * lambda_ + detector_losses_reverse['query_loss'] * (1 - lambda_)
        else:
            # RoI
            # ROI inputs are post_nms_top_k proposals.
            # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

            if self.use_mixup:
                _, detector_losses_reverse = self.roi_heads(images, features, proposals_reverse, gt_instances_reverse)
                detector_losses['loss_box_reg'] = detector_losses['loss_box_reg'] * lambda_ + detector_losses_reverse['loss_box_reg'] * (1-lambda_)
                detector_losses['loss_cls'] = detector_losses['loss_cls'] * lambda_ + detector_losses_reverse['loss_cls'] * (1-lambda_)

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
class MyRCNN(nn.Module):
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
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml


        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std



        self.to(self.device)

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

        if cfg.MODEL.BACKBONE.FREEZE_FIRST2:
            for p in self.backbone.bottom_up.base_layer:
                p.requires_grad = False
            for p in self.backbone.bottom_up.level0:
                p.requires_grad = False
            for p in self.backbone.bottom_up.level1:
                p.requires_grad = False

            print("froze DLA first 2 module parameters")

        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))






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

        # images_aux = None
        if batched_inputs_aux is not None:
            with torch.no_grad():
                images_aux = self.preprocess_image(batched_inputs_aux)
                # features_aux = self.backbone(images_aux.tensor)
        # backbone FPN
        images, hand_feature_imgs = self.preprocess_image(batched_inputs)
        # images = self.preprocess_image(batched_inputs)
        # start = time.time()
        # hand_feature_imgs = self.get_hand_feature(images.tensor)
        # print(f"a {time.time() - start}")
        features = self.backbone(images.tensor)  # List of L, FPN features

        '1，太慢'

        with torch.no_grad():

            hand_feature_features = self.backbone(hand_feature_imgs)



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

        # # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)


        # features_hand = self.describer(images)
        # feature_hand_pooled = self.roi_heads.roi_pooling(features_hand, proposals)

        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, hand_feature_features,)

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

        images, hand_feature_imgs = self.preprocess_image(batched_inputs)
        # hand_feature_imgs = self.get_hand_feature(images.tensor)
        features = self.backbone(images.tensor)

        hand_feature_features = self.backbone(hand_feature_imgs)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # features_hand = self.describer(images)
            # hand_feature_imgs = self.get_hand_feature(images)
            # features_hand = self.cut_from_img(proposals, hand_feature_imgs)
            # features_hand = self.describer(features_hand)

            # feature_hand_pooled = self.roi_heads.roi_pooling(features_hand, proposals)

            results, _ = self.roi_heads(images, features, proposals, None, hand_feature_features)


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

        scaler = MinMaxScaler()
        # start = time.time()
        # hand_feature_imgs = self.get_hand_feature(images.tensor, scaler)
        image_copy = copy.deepcopy(images.tensor)
        hand_feature_imgs = self.get_hand_feature(image_copy, scaler)

        # print(f"a {time.time() - start}")
        return images, hand_feature_imgs

    # from numba import jit

    def get_hand_feature(self, batched_images, scaler):

        window = torch.Tensor([[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                                [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                [1, 4, 5, 3, 0, 3, 5, 4, 1],
                                [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                [2, 5, 0, -24, -40, -24, 0, 5, 2],
                                [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                [1, 4, 5, 3, 0, 3, 4, 4, 1],
                                [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                [0, 1, 1, 2, 2, 2, 1, 1, 0]]])
        window_size = 9


        # batched_images：[b, c, h, w]
        for i in range(batched_images.shape[0]):
            temp = batched_images[i][0, :, :]
            # LBP
            # lbp_image = circular_LBP(temp.cpu().numpy(), 3, 8)
            lbp_image = local_binary_pattern(temp.cpu().numpy(), 3, 8, method="nri_uniform")
            lbp_image = scaler.fit_transform(lbp_image)
            # HoG
            # hog_image = hog1(batched_images[i][0].cpu().numpy(), orientations=6, pixels_per_cell=(16, 16),
            #                 cells_per_block=(4, 4), visualize=True, multichannel=False)
            _, hog_image = hog(batched_images[i][0].cpu().numpy(), orientations=6, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True, multichannel=False)
            hog_image = scaler.fit_transform(hog_image)
            # LoG
            log_image = LoG(temp.cpu(), window, window_size, "L")
            batched_images[i] = torch.cat([torch.tensor(lbp_image, dtype=torch.float32, device="cuda:0").unsqueeze(0),
                                           torch.tensor(log_image, dtype=torch.float32, device="cuda:0"),
                                           torch.tensor(hog_image, dtype=torch.float32, device="cuda:0").unsqueeze(0)], dim=0)

        return batched_images

# https://github.com/PeizeSun/SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/detector.py
@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

#########################
# @jit(nopython=True)
# def _hog_normalize_block1(block, method, eps=1e-5):
#     if method == 'L1':
#         out = block / (np.sum(np.abs(block)) + eps)
#     elif method == 'L1-sqrt':
#         out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
#     elif method == 'L2':
#         out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
#     elif method == 'L2-Hys':
#         out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
#         out = np.minimum(out, 0.2)
#         out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
#     else:
#         raise ValueError('Selected block normalization method is invalid.')
#
#     return out
#
# @jit(nopython=True)
# def _hog_channel_gradient1(channel):
#     """Compute unnormalized gradient image along `row` and `col` axes.
#
#     Parameters
#     ----------
#     channel : (M, N) ndarray
#         Grayscale image or one of image channel.
#
#     Returns
#     -------
#     g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
#     """
#     g_row = np.empty(channel.shape, dtype=np.double)
#     g_row[0, :] = 0
#     g_row[-1, :] = 0
#     g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
#     g_col = np.empty(channel.shape, dtype=np.double)
#     g_col[:, 0] = 0
#     g_col[:, -1] = 0
#     g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
#
#     return g_row, g_col
#
# from skimage import draw
# # @jit([
# #     "float32[:,:](int64, int64, int32[:], float64[:], float64[:], int64, int64, float64[:,:], float64[:,:,:])"
# # ], nopython=True)
# def compute(n_cells_row, n_cells_col, orientations_arr, dr_arr, dc_arr, c_row, c_col, hog_image, orientation_histogram):
#     for r in range(n_cells_row):
#         for c in range(n_cells_col):
#             for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
#                 centre = [r * c_row + c_row // 2,
#                                 c * c_col + c_col // 2]
#                 rr, cc = draw.line(int(centre[0] - dc), int(centre[1] + dr), int(centre[0] + dc), int(centre[1] - dr))
#                 # rr, cc = draw_line(centre, dr, dc)
#                 hog_image[rr, cc] += orientation_histogram[r, c, o]
#
#     return hog_image
#
# # @jit([
# #     "(int64, int64)(int32[:,:], float64, float64)"
# # ], nopython=True)
# # def draw_line(centre, dr, dc):
# #     return draw.line(int(centre[0] - dc), int(centre[1] + dr), int(centre[0] + dc), int(centre[1] - dr))
#
#
# def hog1(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
#         block_norm='L2-Hys', visualize=False, transform_sqrt=False,
#         feature_vector=True, multichannel=None):
#     """Extract Histogram of Oriented Gradients (HOG) for a given image.
#
#     Compute a Histogram of Oriented Gradients (HOG) by
#
#         1. (optional) global image normalization
#         2. computing the gradient image in `row` and `col`
#         3. computing gradient histograms
#         4. normalizing across blocks
#         5. flattening into a feature vector
#
#     Parameters
#     ----------
#     image : (M, N[, C]) ndarray
#         Input image.
#     orientations : int, optional
#         Number of orientation bins.
#     pixels_per_cell : 2-tuple (int, int), optional
#         Size (in pixels) of a cell.
#     cells_per_block : 2-tuple (int, int), optional
#         Number of cells in each block.
#     block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}, optional
#         Block normalization method:
#
#         ``L1``
#            Normalization using L1-norm.
#         ``L1-sqrt``
#            Normalization using L1-norm, followed by square root.
#         ``L2``
#            Normalization using L2-norm.
#         ``L2-Hys``
#            Normalization using L2-norm, followed by limiting the
#            maximum values to 0.2 (`Hys` stands for `hysteresis`) and
#            renormalization using L2-norm. (default)
#            For details, see [3]_, [4]_.
#
#     visualize : bool, optional
#         Also return an image of the HOG.  For each cell and orientation bin,
#         the image contains a line segment that is centered at the cell center,
#         is perpendicular to the midpoint of the range of angles spanned by the
#         orientation bin, and has intensity proportional to the corresponding
#         histogram value.
#     transform_sqrt : bool, optional
#         Apply power law compression to normalize the image before
#         processing. DO NOT use this if the image contains negative
#         values. Also see `notes` section below.
#     feature_vector : bool, optional
#         Return the data as a feature vector by calling .ravel() on the result
#         just before returning.
#     multichannel : boolean, optional
#         If True, the last `image` dimension is considered as a color channel,
#         otherwise as spatial.
#
#     Returns
#     -------
#     out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray
#         HOG descriptor for the image. If `feature_vector` is True, a 1D
#         (flattened) array is returned.
#     hog_image : (M, N) ndarray, optional
#         A visualisation of the HOG image. Only provided if `visualize` is True.
#
#     References
#     ----------
#     .. [1] https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
#
#     .. [2] Dalal, N and Triggs, B, Histograms of Oriented Gradients for
#            Human Detection, IEEE Computer Society Conference on Computer
#            Vision and Pattern Recognition 2005 San Diego, CA, USA,
#            https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf,
#            :DOI:`10.1109/CVPR.2005.177`
#
#     .. [3] Lowe, D.G., Distinctive image features from scale-invatiant
#            keypoints, International Journal of Computer Vision (2004) 60: 91,
#            http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf,
#            :DOI:`10.1023/B:VISI.0000029664.99615.94`
#
#     .. [4] Dalal, N, Finding People in Images and Videos,
#            Human-Computer Interaction [cs.HC], Institut National Polytechnique
#            de Grenoble - INPG, 2006,
#            https://tel.archives-ouvertes.fr/tel-00390303/file/NavneetDalalThesis.pdf
#
#     Notes
#     -----
#     The presented code implements the HOG extraction method from [2]_ with
#     the following changes: (I) blocks of (3, 3) cells are used ((2, 2) in the
#     paper); (II) no smoothing within cells (Gaussian spatial window with sigma=8pix
#     in the paper); (III) L1 block normalization is used (L2-Hys in the paper).
#
#     Power law compression, also known as Gamma correction, is used to reduce
#     the effects of shadowing and illumination variations. The compression makes
#     the dark regions lighter. When the kwarg `transform_sqrt` is set to
#     ``True``, the function computes the square root of each color channel
#     and then applies the hog algorithm to the image.
#     """
#     image = np.atleast_2d(image)
#
#     if multichannel is None:
#         multichannel = (image.ndim == 3)
#
#     ndim_spatial = image.ndim - 1 if multichannel else image.ndim
#     if ndim_spatial != 2:
#         raise ValueError('Only images with 2 spatial dimensions are '
#                          'supported. If using with color/multichannel '
#                          'images, specify `multichannel=True`.')
#
#     if transform_sqrt:
#         image = np.sqrt(image)
#
#     if image.dtype.kind == 'u':
#         # convert uint image to float
#         # to avoid problems with subtracting unsigned numbers
#         image = image.astype('float')
#
#     if multichannel:
#         g_row_by_ch = np.empty_like(image, dtype=np.double)
#         g_col_by_ch = np.empty_like(image, dtype=np.double)
#         g_magn = np.empty_like(image, dtype=np.double)
#
#         for idx_ch in range(image.shape[2]):
#             g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
#                 _hog_channel_gradient1(image[:, :, idx_ch])
#             g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
#                                             g_col_by_ch[:, :, idx_ch])
#
#         # For each pixel select the channel with the highest gradient magnitude
#         idcs_max = g_magn.argmax(axis=2)
#         rr, cc = np.meshgrid(np.arange(image.shape[0]),
#                              np.arange(image.shape[1]),
#                              indexing='ij',
#                              sparse=True)
#         g_row = g_row_by_ch[rr, cc, idcs_max]
#         g_col = g_col_by_ch[rr, cc, idcs_max]
#     else:
#         g_row, g_col = _hog_channel_gradient1(image)
#
#     s_row, s_col = image.shape[:2]
#     c_row, c_col = pixels_per_cell
#     b_row, b_col = cells_per_block
#
#     n_cells_row = int(s_row // c_row)  # number of cells along row-axis
#     n_cells_col = int(s_col // c_col)  # number of cells along col-axis
#
#     # compute orientations integral images
#     orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))
#
#     _hoghistogram.hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
#                                  n_cells_col, n_cells_row,
#                                  orientations, orientation_histogram)
#
#     # now compute the histogram for each cell
#     hog_image = None
#
#     if visualize:
#
#
#
#
#
#         radius = min(c_row, c_col) // 2 - 1
#         orientations_arr = np.arange(orientations)
#         # set dr_arr, dc_arr to correspond to midpoints of orientation bins
#         orientation_bin_midpoints = (
#             np.pi * (orientations_arr + .5) / orientations)
#         dr_arr = radius * np.sin(orientation_bin_midpoints)
#         dc_arr = radius * np.cos(orientation_bin_midpoints)
#         hog_image = np.zeros((s_row, s_col), dtype=float)
#
#
#
#
#         hog_image = compute(n_cells_row, n_cells_col, orientations_arr, dr_arr, dc_arr, c_row, c_col, hog_image, orientation_histogram)
#
#     # n_blocks_row = (n_cells_row - b_row) + 1
#     # n_blocks_col = (n_cells_col - b_col) + 1
#     # normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,
#     #                               b_row, b_col, orientations))
#     #
#     # for r in range(n_blocks_row):
#     #     for c in range(n_blocks_col):
#     #         block = orientation_histogram[r:r + b_row, c:c + b_col, :]
#     #         normalized_blocks[r, c, :] = \
#     #             _hog_normalize_block1(block, method=block_norm)
#     #
#     # """
#     # The final step collects the HOG descriptors from all blocks of a dense
#     # overlapping grid of blocks covering the detection window into a combined
#     # feature vector for use in the window classifier.
#     # """
#     #
#     # if feature_vector:
#     #     normalized_blocks = normalized_blocks.ravel()
#
#     if visualize:
#         return hog_image
#     else:
#         return None
#############################



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
