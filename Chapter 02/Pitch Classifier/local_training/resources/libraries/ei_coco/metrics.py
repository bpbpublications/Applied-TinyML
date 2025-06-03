# ported from keras-cv
# keras_cv/metrics/coco/pycoco_wrapper.py r0.5.0

# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
from typing import List

from .conversion import convert_y_true_boundingbox_label_scores_to_coco_groundtruth
from .conversion import convert_y_pred_boundingbox_label_scores_to_coco_detections

from ei_shared.labels import BoundingBoxLabelScore

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "ARmax1",
    "ARmax10",
    "ARmax100",
    "ARs",
    "ARm",
    "ARl",
]

METRIC_MAPPING = {
    "AP": "MaP",
    "AP50": "MaP@[IoU=50]",
    "AP75": "MaP@[IoU=75]",
    "APs": "MaP@[area=small]",
    "APm": "MaP@[area=medium]",
    "APl": "MaP@[area=large]",
    "ARmax1": "Recall@[max_detections=1]",
    "ARmax10": "Recall@[max_detections=10]",
    "ARmax100": "Recall@[max_detections=100]",
    "ARs": "Recall@[area=small]",
    "ARm": "Recall@[area=medium]",
    "ARl": "Recall@[area=large]",
}


class HidePrints:
    """A basic internal only context manager to hide print statements."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def calculate_coco_metrics(
    y_true_bbox_labels: List[List[BoundingBoxLabelScore]],
    y_pred_bbox_labels: List[List[BoundingBoxLabelScore]],
    width: int,
    height: int,
    num_classes: int,
):
    """Calculate the full suite of coco metrics provided by pycocotools

    Args:
        y_true_bbox_labels: ground truth values contained bounding boxes and labels
        y_pred_bbox_labels: bounding box predictions.
        width: input image width
        height: input image height
        num_classes: total number of classes
    Returns:
        a dict containing the pycocotools.coco_eval metrics
    """

    # Late import this, since 'pycocotools' is not installed on all
    # containers that depend on this file.
    from pycocotools.cocoeval import COCOeval
    from .pycoco_wrapper import PyCOCOWrapper

    coco_gt_dataset = convert_y_true_boundingbox_label_scores_to_coco_groundtruth(
        y_true_bbox_labels, width, height, num_classes
    )
    coco_predictions = convert_y_pred_boundingbox_label_scores_to_coco_detections(
        y_pred_bbox_labels, width, height
    )

    with HidePrints():
        coco_gt = PyCOCOWrapper(gt_dataset=coco_gt_dataset)
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)

        image_ids = [ann["image_id"] for ann in coco_predictions]

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats

    metrics_dict = {}
    for i, name in enumerate(METRIC_NAMES):
        metrics_dict[METRIC_MAPPING[name]] = float(coco_metrics[i])

    metrics_dict["support"] = {
        "images": len(coco_gt_dataset["images"]),
        "annotations": len(coco_gt_dataset["annotations"]),
        "detections": len(coco_predictions),
    }

    return metrics_dict
