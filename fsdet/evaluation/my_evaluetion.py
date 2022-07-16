import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import fsdet.utils.comm as comm
from fsdet.data import MetadataCatalog
from fsdet.data.datasets.coco import convert_to_coco_json
from fsdet.structures import BoxMode
from fsdet.utils.logger import create_small_table

from .evaluator import DatasetEvaluator


class NewDatasetEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1, 2, 3, 4, 5
        ]
        self._novel_classes = [1, 2, 3, 4, 5]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        self.plot_pr = cfg.TEST.PLOT_PR.ENABLED
        self.conf = cfg.TEST.PLOT_PR.PLOT_PR_CONF
        self.threshold = cfg.TEST.PLOT_PR.PLOT_PR_THRESHOLD
        self.class_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            if self.plot_pr:
                coco_eval = (
                    _evaluate_predictions_on_coco_and_plot(
                        self._coco_api, self._coco_results, "bbox", output_dir=self._output_dir,
                        class_number=self.class_num, conf=self.conf, threshold=self.threshold,
                        plot=self.plot_pr, class_name=self._metadata.get("thing_classes")
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            else:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox",
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            precision_50 = precisions[:, :, idx, 0, -1][0]
            precision_50 = precision_50[precision_50 > -1]
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # tabulate it
        N_COLS = min(6, len(results_per_category_50) * 2)
        results_flatten = list(itertools.chain(*results_per_category_50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class NewDatasetEvaluator2(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1, 2, 3, 4
        ]
        self._novel_classes = [1, 2, 3, 4]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class NewDatasetEvaluator3(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1, 2, 3, 4, 5
        ]
        self._novel_classes = [1, 2, 3, 4, 5]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            precision_50 = precisions[:, :, idx, 0, -1][0]
            precision_50 = precision_50[precision_50 > -1]
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # tabulate it
        N_COLS = min(6, len(results_per_category_50) * 2)
        results_flatten = list(itertools.chain(*results_per_category_50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class NewDatasetEvaluator4(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1, 2, 3, 4, 5
        ]
        self._novel_classes = [1, 2]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            precision_50 = precisions[:, :, idx, 0, -1][0]
            precision_50 = precision_50[precision_50 > -1]
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # tabulate it
        N_COLS = min(6, len(results_per_category_50) * 2)
        results_flatten = list(itertools.chain(*results_per_category_50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class NewDatasetEvaluator5(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
        ]
        self._novel_classes = []

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            precision_50 = precisions[:, :, idx, 0, -1][0]
            precision_50 = precision_50[precision_50 > -1]
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # tabulate it
        N_COLS = min(6, len(results_per_category_50) * 2)
        results_flatten = list(itertools.chain(*results_per_category_50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class NewDatasetEvaluator6(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):  # initial needed variables
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
                          or "novel" in dataset_name
        self._base_classes = [
            1
        ]
        self._novel_classes = [1, 2]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        self.plot_pr = cfg.TEST.PLOT_PR.ENABLED
        self.conf = cfg.TEST.PLOT_PR.PLOT_PR_CONF
        self.threshold = cfg.TEST.PLOT_PR.PLOT_PR_THRESHOLD
        self.class_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            # coco_eval = (
            #     _evaluate_predictions_on_coco_and_plot(
            #         self._coco_api, self._coco_results, "bbox", output_dir=self._output_dir,
            #                         class_number=self.class_num, conf=self.conf, threshold=self.threshold,
            #                         plot=self.plot_pr, class_name=self._metadata.get("thing_classes")
            #     )
            #     if len(self._coco_results) > 0
            #     else None  # cocoapi does not handle empty results very well
            # )
            '重写了CoCo的接口'
            coco_eval = (
                my_evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox", output_dir=self._output_dir,
                        class_number=self.class_num, conf=self.conf, threshold=self.threshold,
                        plot=self.plot_pr, class_name=self._metadata.get("thing_classes")
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        # metrics = ["AP", "AP10", "AP20", "AP30", "AP40", "AP50", "AP75", "APs", "APm", "APl"]
        metrics = ["AP", "AP30", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            precision_50 = precisions[:, :, idx, 0, -1][0]
            precision_50 = precision_50[precision_50 > -1]
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # tabulate it
        N_COLS = min(6, len(results_per_category_50) * 2)
        results_flatten = list(itertools.chain(*results_per_category_50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def _evaluate_predictions_on_coco_and_plot(coco_gt, coco_results, iou_type, catIds=None, output_dir="", class_number=0,
                                           conf=0.5, threshold=0.5, plot=False, class_name=()):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)

    # https://blog.csdn.net/weixin_42899627/article/details/120578787
    if plot:
        from .confusion_matrix import ConfusionMatrix, xywh2xyxy, process_batch, ap_per_class
        C_M = ConfusionMatrix(number_class=class_number, conf=conf, iou_thres=threshold)
        stats = []
        for i, _ in coco_gt.imgs.items():

            # for i in range(len(coco_gt.imgs)):  # 460张图
            bbox_gt = np.array([y['bbox'] for y in coco_gt.imgToAnns[i]])
            class_gt = np.array([[y['category_id'] - 1] for y in coco_gt.imgToAnns[i]])
            labels = np.hstack((class_gt, bbox_gt))

            bbox_dt = np.array([y['bbox'] for y in coco_dt.imgToAnns[i]])
            conf_dt = np.array([[y['score']] for y in coco_dt.imgToAnns[i]])
            class_dt = np.array([[y['category_id'] - 1] for y in coco_dt.imgToAnns[i]])
            predictions = np.hstack((np.hstack((bbox_dt, conf_dt)), class_dt))

            if len(predictions) == 0:
                continue

            C_M.process_batch(predictions, labels)

            '''PR等曲线'''
            detects = torch.tensor(xywh2xyxy(predictions))
            labs = torch.tensor(np.hstack((labels[:, 0][:, None], xywh2xyxy(labels[:, 1:]))))
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            correct = process_batch(detects, labs, iouv)
            tcls = labs[:, 0].tolist()  # target class
            stats.append((correct.cpu(), detects[:, 4].cpu(), detects[:, 5].cpu(), tcls))

        C_M.print()

        plot_dir = output_dir

        names = {k: v for k, v in enumerate(class_name)}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=plot_dir, names=names)
        C_M.plot(save_dir=plot_dir + 'confusion_matrix_rec.png', names=class_name, rec_or_pred=0)
        C_M.plot(save_dir=plot_dir + 'confusion_matrix_pred.png', names=class_name, rec_or_pred=1)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


from .my_cocoeval import MyCOCOeval
def my_evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None, output_dir="", class_number=0,
                                           conf=0.5, threshold=0.5, plot=False, class_name=()):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    if plot:
        from .confusion_matrix import ConfusionMatrix, xywh2xyxy, process_batch, ap_per_class
        C_M = ConfusionMatrix(number_class=class_number, conf=conf, iou_thres=threshold)
        stats = []
        for i, _ in coco_gt.imgs.items():
            # for i in range(len(coco_gt.imgs)):  # 460张图
            bbox_gt = np.array([y['bbox'] for y in coco_gt.imgToAnns[i]])
            class_gt = np.array([[y['category_id'] - 1] for y in coco_gt.imgToAnns[i]])
            labels = np.hstack((class_gt, bbox_gt))
            if len(class_gt) == 0:
                continue


            bbox_dt = np.array([y['bbox'] for y in coco_dt.imgToAnns[i]])
            conf_dt = np.array([[y['score']] for y in coco_dt.imgToAnns[i]])
            class_dt = np.array([[y['category_id'] - 1] for y in coco_dt.imgToAnns[i]])
            predictions = np.hstack((np.hstack((bbox_dt, conf_dt)), class_dt))

            C_M.process_batch(predictions, labels)

            '''PR等曲线'''
            detects = torch.tensor(xywh2xyxy(predictions))
            labs = torch.tensor(np.hstack((labels[:, 0][:, None], xywh2xyxy(labels[:, 1:]))))
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            # iouv = np.linspace(.3, 0.95, int(np.round((0.95 - .3) / .05)) + 1, endpoint=True)
            correct = process_batch(detects, labs, iouv)
            tcls = labs[:, 0].tolist()  # target class
            stats.append((correct.cpu(), detects[:, 4].cpu(), detects[:, 5].cpu(), tcls))

        C_M.print()

        plot_dir = output_dir

        names = {k: v for k, v in enumerate(class_name)}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=plot_dir, names=names)
        C_M.plot(normalize=False, save_dir=plot_dir + 'confusion_matrix_rec.png', names=["targrt"], rec_or_pred=0)
        C_M.plot(normalize=False, save_dir=plot_dir + 'confusion_matrix_pred.png', names=["targrt"], rec_or_pred=1)



    coco_eval = MyCOCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
