import numpy as np
from sklearn import metrics as sklearn_metrics
import warnings
from typing import List, Any, Optional

import ei_tensorflow.constrained_object_detection.metrics as fomo_metrics
from ei_shared.metrics_utils import calculate_grouped_metrics
from ei_coco.metrics import calculate_coco_metrics
from ei_shared.labels import BoundingBoxLabelScore


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[List] = None,
    max_groups: Optional[int] = None,
    # Don't include "all" in the grouped metrics by default because there's often no need
    groups_include_all: bool = False,
):
    """Calculate a collection of regression specific metrics.

    Args:
        y_true: ground truth values
        y_pred: predictions
        groups: (optional) grouping of N elements for y_true, y_pred.
        max_groups: (optional) if set, and groups provided, only use top max_groups by frequency.
    Returns:
        a dict containing a collection of sklearn regression metrics.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    def _calc_metrics(yt, yp):
        return {
            "mean_squared_error": sklearn_metrics.mean_squared_error(yt, yp),
            "mean_absolute_error": sklearn_metrics.mean_absolute_error(yt, yp),
            "explained_variance_score": sklearn_metrics.explained_variance_score(
                yt, yp
            ),
            "support": len(yt),
        }

    if groups is None:
        return _calc_metrics(y_true, y_pred)
    else:
        return calculate_grouped_metrics(
            y_true,
            y_pred,
            _calc_metrics,
            groups,
            max_groups,
            include_all=groups_include_all,
        )


def can_ignore_roc_auc_score_exception(e: Exception):
    """Return True if an exception thrown by sklearn.roc_auc_score can be ignored.

    sklearn_metrics.roc_auc_score has a number of cases where we can ignore
    the exception and just mark the metric, as effectively, not applicable.
    """

    ignorable_sub_strings = [
        "Number of classes in y_true not equal to the number of columns in 'y_score'",
        "Only one class present in y_true",
    ]
    for sub_string in ignorable_sub_strings:
        if sub_string in str(e):
            return True
    return False


def calculate_classification_metrics(
    y_true_one_hot: np.ndarray,
    y_pred_probs: np.ndarray,
    num_classes: int,
    groups: Optional[List] = None,
    max_groups: Optional[int] = None,
    # Don't include "all" in the grouped metrics by default because there's often no need
    groups_include_all: bool = False,
):
    """Calculate a collection of classification specific metrics.

    Args:
        y_true_one_hot: ground truth values in one hot format.
        y_pred_probs: predictions containing full probability distribution
        num_classes: total number of classes
        groups: (optional) grouping of N elements for y_true, y_pred.
        max_groups: (optional) if set, and groups provided, only use top max_groups by frequency.
    Returns:
        a dict containing a collection of sklearn classification metrics.
    """

    # TODO: derive num_classes from width of y_true_one_hot?

    # sanity check y_true_one_hot look one hot
    # or at least close to one hot, since for int8 model values
    # they end up being a tad off 1.0 :/
    row_wise_sums = np.sum(y_true_one_hot, axis=-1)
    difference_from_1 = np.abs(1 - row_wise_sums)
    if difference_from_1.max() > 1e-2:
        print(
            f"WARNING: y_true_one_hot provided does not look one hot {difference_from_1.max()}"
        )

    # always renormalise y_pred_probs. we do this int8 values can
    # give a result that fails the atol tests of roc_auc
    y_pred_probs /= y_pred_probs.sum(axis=-1, keepdims=True)

    # build a labels list, [0, 1, 2, ...] which is used by
    # a number of the sklearn metrics
    labels = list(range(num_classes))

    def _calc_metrics(y_true_one_hot, y_pred_probs):
        # convert from distribution to labels for some metrics
        y_true_labels = y_true_one_hot.argmax(axis=-1)
        y_pred_labels = y_pred_probs.argmax(axis=-1)

        metrics: dict[str, Any] = {}

        metrics["confusion_matrix"] = sklearn_metrics.confusion_matrix(
            y_true_labels, y_pred_labels, labels=labels
        ).tolist()

        metrics["classification_report"] = sklearn_metrics.classification_report(
            y_true_labels, y_pred_labels, output_dict=True, zero_division=0
        )

        try:
            if num_classes == 2:
                # NOTE! roc_auc calculation for binary case must be called with
                #       labels otherwise it throws an exception
                metrics["roc_auc"] = sklearn_metrics.roc_auc_score(
                    y_true=y_true_labels, y_score=y_pred_labels, multi_class="ovr"
                )
            else:
                metrics["roc_auc"] = sklearn_metrics.roc_auc_score(
                    y_true=y_true_labels, y_score=y_pred_probs, multi_class="ovr"
                )
        except Exception as e:
            # a known common case for this is when not all classes are
            # represented. we can detect this from the exception and ignore. but
            # if it's something else we should reraise
            if can_ignore_roc_auc_score_exception(e):
                metrics["roc_auc"] = None
            else:
                raise e

        metrics["loss"] = sklearn_metrics.log_loss(
            y_true_labels, y_pred_probs, labels=labels
        )

        metrics["support"] = len(y_true_labels)

        # copy the weighted average P/R/F1 scores out of the classication
        # report for UI
        metrics["weighted_average"] = {}
        for key in ["precision", "recall", "f1-score"]:
            metrics["weighted_average"][key] = metrics["classification_report"][
                "weighted avg"
            ][key]

        return metrics

    if groups is None:
        return _calc_metrics(y_true_one_hot, y_pred_probs)
    else:
        return calculate_grouped_metrics(
            y_true_one_hot,
            y_pred_probs,
            _calc_metrics,
            groups,
            max_groups,
            include_all=groups_include_all,
        )


def _coco_map_calculation_from_studio(
    y_true_bbox_labels: List[List[BoundingBoxLabelScore]],
    prediction: List[List[BoundingBoxLabelScore]],
    num_classes: int,
):

    # coco map calculation taken as is from ei_tensorflow.profiling.
    # keep as seperate method to denote code copied as is

    # This is only installed on object detection containers so import it only when used
    from mean_average_precision import MetricBuilder

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=num_classes
    )

    if len(y_true_bbox_labels) != len(prediction):
        raise Exception(
            "Expected to have same number of y_true and y_pred"
            f" but was {len(y_true_bbox_labels)}"
            f" and {len(prediction)}"
        )

    for instance_y_true, instance_y_pred in zip(y_true_bbox_labels, prediction):
        gt = []
        curr_ps = []

        for bbox_label_score in instance_y_true:
            bbox = bbox_label_score.bbox
            # The library expects [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            gt.append(
                [bbox.x0, bbox.y0, bbox.x1, bbox.y1, bbox_label_score.label, 0, 0]
            )

        for bbox_label_score in instance_y_pred:
            bbox = bbox_label_score.bbox
            # The library expects [xmin, ymin, xmax, ymax, class_id, confidence]
            curr_ps.append(
                [
                    bbox.x0,
                    bbox.y0,
                    bbox.x1,
                    bbox.y1,
                    bbox_label_score.label,
                    bbox_label_score.score,
                ]
            )

        gt = np.array(gt)
        curr_ps = np.array(curr_ps)
        metric_fn.add(curr_ps, gt)

    coco_map = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )["mAP"]

    return float(coco_map)


def calculate_object_detection_metrics(
    y_true_bbox_labels: List[List[BoundingBoxLabelScore]],
    y_pred_bbox_labels: List[List[BoundingBoxLabelScore]],
    width: int,
    height: int,
    num_classes: int,
    groups: Optional[List] = None,
    max_groups: Optional[int] = None,
    # Don't include "all" in the grouped metrics by default because there's often no need
    groups_include_all: bool = False,
):
    """Calculate a collection of object detection specific metrics.

    Args:
        y_true_bbox_labels: ground truth values contained bounding boxes and labels
        y_pred: bounding box predictions.
        num_classes: total number of classes
        groups: (optional) grouping of N elements for y_true, y_pred.
        max_groups: (optional) if set, and groups provided, only use top max_groups by frequency.
    Returns:
        a dict containing a collection of sklearn object detection metrics.
    """

    def _calc_metrics(y_true, y_pred):
        metrics = {}

        metrics["support"] = len(y_true)

        metrics["coco_metrics"] = calculate_coco_metrics(
            y_true_bbox_labels=y_true_bbox_labels,
            y_pred_bbox_labels=y_pred_bbox_labels,
            width=width,
            height=height,
            num_classes=num_classes,
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            metrics["coco_map"] = _coco_map_calculation_from_studio(
                num_classes=num_classes,
                y_true_bbox_labels=y_true,
                prediction=y_pred,
            )

        return metrics

    if groups is None:
        return _calc_metrics(y_true_bbox_labels, y_pred_bbox_labels)
    else:
        return calculate_grouped_metrics(
            y_true_bbox_labels,
            y_pred_bbox_labels,
            _calc_metrics,
            groups,
            max_groups,
            include_all=groups_include_all,
        )


def calculate_fomo_metrics(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    num_classes: int,
    groups: Optional[List] = None,
    max_groups: Optional[int] = None,
    # Don't include "all" in the grouped metrics by default because there's often no need
    groups_include_all: bool = False,
):
    """Calculate a collection of FOMO specific metrics.
    These are calculated seperately from object detection because, fundamentally,
    FOMO is more like NxN classification and segmentation than bounding box
    detection.

    Args:
        y_true_labels: ground truth labels with same shape as y_pred_labels
        y_pred_labels: predicted labels with same shape as y_true_labels
        num_classes: total number of classes.
        groups: (optional) grouping of N elements for y_true, y_pred.
        max_groups: (optional) if set, and groups provided, only use top max_groups by frequency.
    Returns:
        a dict containing a collection of sklearn & fomo specific detection metrics.
    """

    if y_true_labels.shape != y_pred_labels.shape:
        raise Exception(
            "Expected y_true_labels and y_pred_labels to be the same"
            " shape, and both (N,W,H) but they were shaped"
            f" {y_true_labels.shape} and {y_pred_labels.shape}"
        )

    def _calc_metrics(y_true, y_pred):
        metrics = {}

        metrics["support"] = len(y_true)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        metrics["confusion_matrix"] = sklearn_metrics.confusion_matrix(
            y_true, y_pred, labels=range(num_classes)
        ).tolist()

        precision, recall, f1 = fomo_metrics.non_background_metrics(
            y_true, y_pred, num_classes
        )
        metrics["non_background"] = {"precision": precision, "recall": recall, "f1": f1}

        metrics["classification_report"] = sklearn_metrics.classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        return metrics

    if groups is None:
        return _calc_metrics(y_true_labels, y_pred_labels)
    else:
        return calculate_grouped_metrics(
            y_true_labels,
            y_pred_labels,
            _calc_metrics,
            groups,
            max_groups,
            include_all=groups_include_all,
        )
