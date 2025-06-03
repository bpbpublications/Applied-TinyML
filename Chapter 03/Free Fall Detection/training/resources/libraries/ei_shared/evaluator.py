import json
import numpy as np
import traceback
from typing import List, Any, Optional, Dict, Union
from itertools import chain

import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
import sklearn.metrics

from ei_shared.labels import BoundingBoxLabelScore
from ei_shared.facetted_metrics import FacettedMetrics
from ei_shared.metrics_utils import quantize_metadata
from ei_coco.metrics import calculate_coco_metrics

from ei_sklearn.metrics import calculate_regression_metrics
from ei_sklearn.metrics import calculate_classification_metrics
from ei_sklearn.metrics import calculate_object_detection_metrics
from ei_sklearn.metrics import calculate_fomo_metrics

from ei_tensorflow.constrained_object_detection import models


def ei_log(msg: str):
    print("EI_LOG_LEVEL=debug", msg)


class EvalResult:
    """Represents the results of evaluating a model, including defaults to be written to the model metadata file."""

    # TODO: These default values are historical. We should figure out if they are needed.
    def __init__(
        self, metrics: Dict[str, Any] = {}, matrix=[], report={}, accuracy=0, loss=0
    ):
        # A dictionary including many metrics
        self.metrics = metrics
        self.matrix = matrix
        self.report = report
        self.accuracy = accuracy
        self.loss = loss


class Evaluator:
    """
    Class to evaluate the performance of a model on a dataset, including subgroup metrics.
    """

    def __init__(
        self,
        per_sample_metadata: Optional[Dict[int, Dict[str, Union[str, int, float]]]] = None,
        row_to_sample_id: Optional[List[int]] = None,
    ):
        """
        Constructs an Evaluator object. If metadata is provided it will quantize the metadata so
        that it can be used for calculating subgroup metrics.
        Args:
            per_sample_metadata: dictionary mapping sample_id to dictionary
                                of { meta_data_key: meta_data_value, ... }.
            row_to_sample_id: list of sample IDs in the order they appear in the dataset.
        """
        if per_sample_metadata is None:
            ei_log(
                "Not calculating subgroup metrics because there is no per sample metadata."
            )
            per_sample_metadata = {}
        if row_to_sample_id is None:
            ei_log(
                "Not calculating subgroup metrics because there is no row to sample ID mapping."
            )
            row_to_sample_id = []

        self._per_sample_metadata = per_sample_metadata
        self._row_to_sample_id = row_to_sample_id

        self._quantized_metadata = self._prepare_metadata_for_facetted_metrics()

    def regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Provides statistics on regression performance given labels and predictions.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        result = EvalResult()
        try:
            # Place these in the expected structure
            result.metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)
            result.loss = result.metrics["mean_squared_error"]

            subgroup_metrics = self._calculate_subgroup_metrics_regression(
                y_true=y_true,
                y_pred=y_pred,
            )

            if subgroup_metrics is not None:
                result.metrics["subgroup_metrics"] = subgroup_metrics

            ei_log(f"eval_regression {json.dumps(result.metrics)}")
        except Exception:
            print(
                "Error while calculating metrics. Some metrics may not be available.",
                flush=True,
            )
            # Print the full exception even though we're catching it
            print(traceback.format_exc(), flush=True)

        return result

    def classification(
        self,
        y_true_one_hot: np.ndarray,
        y_pred_probs: np.ndarray,
        class_names: List[str],
    ):
        """
        Provides statistics on classification performance given labels and predictions.

        Args:
            y_true_one_hot: True values in one-hot encoding
            y_pred_probs: Predicted probabilities
            class_names: List of class names
        """
        result = EvalResult()
        try:
            # Place these in the expected structure
            result.metrics = calculate_classification_metrics(
                y_true_one_hot=y_true_one_hot,
                y_pred_probs=y_pred_probs,
                num_classes=len(class_names),
            )
            result.matrix = result.metrics["confusion_matrix"]
            result.report = result.metrics["classification_report"]
            result.accuracy = result.metrics["classification_report"]["accuracy"]
            result.loss = result.metrics["loss"]

            subgroup_metrics = self._calculate_subgroup_metrics_classification(
                class_names=class_names,
                y_true_one_hot=y_true_one_hot,
                y_true_preds=y_pred_probs,
            )

            if subgroup_metrics is not None:
                result.metrics["subgroup_metrics"] = subgroup_metrics

            result.metrics["class_names"] = class_names

            ei_log(f"eval_classification {json.dumps(result.metrics)}")

        except Exception:
            print(
                "Error while calculating metrics. Some metrics may not be available.",
                flush=True,
            )
            # Print the full exception even though we're catching it
            print(traceback.format_exc(), flush=True)

        return result

    def object_detection(
        self,
        class_names: List[str],
        width: int,
        height: int,
        y_true_bbls: List[List[BoundingBoxLabelScore]],
        y_pred_bbls: List[List[BoundingBoxLabelScore]],
    ):
        """
        Provides statistics on object detection performance given labels and predictions.

        Args:
            class_names: List of class names
            width: Width of the image
            height: Height of the image
            y_true_bbls: True bounding box labels
            y_pred_bbls: Predicted bounding box labels
        """
        result = EvalResult()
        try:
            num_classes = len(class_names)

            result.metrics = calculate_object_detection_metrics(
                y_true_bbox_labels=y_true_bbls,
                y_pred_bbox_labels=y_pred_bbls,
                width=width,
                height=height,
                num_classes=num_classes,
            )
            result.accuracy = result.metrics["coco_map"]
            result.metrics["class_names"] = class_names

            subgroup_metrics = self._calculate_subgroup_metrics_object_detection(
                width=width,
                height=height,
                num_classes=num_classes,
                y_true_bbls=y_true_bbls,
                y_pred_bbls=y_pred_bbls,
            )

            if subgroup_metrics is not None:
                result.metrics["subgroup_metrics"] = subgroup_metrics

            ei_log(f"eval_object_detection {json.dumps(result.metrics)}")

        except Exception:
            print(
                "Error while calculating metrics. Some metrics may not be available.",
                flush=True,
            )
            # Print the full exception even though we're catching it
            print(traceback.format_exc(), flush=True)

        return result

    def fomo(
        self,
        class_names: List[str],
        y_true_labels: np.ndarray,
        y_pred_labels: np.ndarray,
    ):
        """
        Provides statistics on FOMO performance given labels and predictions.

        Args:
            class_names: List of class names
            y_true_labels: True labels
            y_pred_labels: Predicted labels
        """
        result = EvalResult()
        try:

            class_names_with_background = ["_background"] + class_names
            num_classes = len(class_names_with_background)

            result.metrics = calculate_fomo_metrics(
                y_true_labels, y_pred_labels, num_classes
            )
            result.matrix = result.metrics["confusion_matrix"]
            result.report = result.metrics["classification_report"]
            result.accuracy = result.metrics["non_background"]["f1"]
            result.metrics["class_names"] = class_names_with_background

            subgroup_metrics = self._calculate_subgroup_metrics_fomo(
                num_classes=num_classes,
                y_true_labels=y_true_labels,
                y_pred_labels=y_pred_labels,
            )

            if subgroup_metrics is not None:
                result.metrics["subgroup_metrics"] = subgroup_metrics
            ei_log(f"eval_fomo {json.dumps(result.metrics)}")
        except Exception:
            print(
                "Error while calculating metrics. Some metrics may not be available.",
                flush=True,
            )
            # Print the full exception even though we're catching it
            print(traceback.format_exc(), flush=True)

        return result

    def _calculate_subgroup_metrics_classification(
        self,
        class_names: List[str],
        y_true_one_hot: np.ndarray,
        y_true_preds: np.ndarray,
    ):
        """
        Calculate metrics that capture differences between subgroups of samples based on metadata,
        in a classification context.

        Args:
            class_names: List of class names
            y_true_one_hot: True values in one-hot encoding
            y_true_preds: Predicted probabilities
        """
        if self._no_subgroup_information():
            return None

        # Instantiate facetting thing
        def log_loss_argmax(y_true, y_pred):
            # Facetted metrics code expects loss for each item individually
            output = [
                sklearn.metrics.log_loss(
                    [y_true[i].argmax()],
                    [y_pred[i]],
                    labels=[c for c in range(len(class_names))],
                )
                for i in range(len(y_true))
            ]
            return output

        facetted_metrics = FacettedMetrics(
            per_sample_metadata=self._quantized_metadata,
            loss_fn=log_loss_argmax,
            stats_test_name="ttest_ind",
        )

        output = {"facetted": None, "per_key": {}}

        output["facetted"] = facetted_metrics.run_test(
            y_true=y_true_one_hot,
            y_pred=y_true_preds,
            row_to_sample_id=self._row_to_sample_id,
        )

        for key, grouping in self._metadata_key_groupings(facetted_metrics):
            output["per_key"][key] = calculate_classification_metrics(
                y_true_one_hot=y_true_one_hot,
                y_pred_probs=y_true_preds,
                num_classes=len(class_names),
                groups=grouping,
            )

        return output

    def _calculate_subgroup_metrics_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Calculate metrics that capture differences between subgroups of samples based on metadata,
        in a regression context.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        if self._no_subgroup_information():
            return None

        def absolute_error(y_true, y_pred):
            # Expects an individual loss for each item (as opposed to e.g. MAE)
            output = [np.abs(y_true[i] - y_pred[i])[0] for i in range(len(y_true))]
            return output

        facetted_metrics = FacettedMetrics(
            per_sample_metadata=self._quantized_metadata,
            loss_fn=absolute_error,
            stats_test_name="ttest_ind",
        )

        output = {"facetted": None, "per_key": {}}

        output["facetted"] = facetted_metrics.run_test(
            y_true=y_true, y_pred=y_pred, row_to_sample_id=self._row_to_sample_id
        )

        for key, grouping in self._metadata_key_groupings(facetted_metrics):
            output["per_key"][key] = calculate_regression_metrics(
                y_true=y_true,
                y_pred=y_pred,
                groups=grouping,
            )

        return output

    def _calculate_subgroup_metrics_object_detection(
        self,
        width: int,
        height: int,
        num_classes: int,
        y_true_bbls: List[List[BoundingBoxLabelScore]],
        y_pred_bbls: List[List[BoundingBoxLabelScore]],
    ):
        """
        Calculate metrics that capture differences between subgroups of samples based on metadata,
        in an object detection context.

        Args:
            width: Width of the image
            height: Height of the image
            num_classes: Number of classes
            y_true_bbls: True bounding box labels
            y_pred_bbls: Predicted bounding box labels
        """
        if self._no_subgroup_information():
            return None

        def coco_map(
            y_true: List[List[BoundingBoxLabelScore]],
            y_pred: List[List[BoundingBoxLabelScore]],
        ):
            # Facetted metrics code expects loss for each item individually
            return [
                1
                - calculate_coco_metrics(
                    [y_true[i]], [y_pred[i]], width, height, num_classes
                )["MaP"]
                for i in range(len(y_true))
            ]

        facetted_metrics = FacettedMetrics(
            per_sample_metadata=self._quantized_metadata,
            loss_fn=coco_map,
            stats_test_name="ttest_ind",
        )

        output = {"facetted": None, "per_key": {}}

        output["facetted"] = facetted_metrics.run_test(
            y_true=y_true_bbls,
            y_pred=y_pred_bbls,
            row_to_sample_id=self._row_to_sample_id,
        )

        for key, grouping in self._metadata_key_groupings(facetted_metrics):
            output["per_key"][key] = calculate_object_detection_metrics(
                y_true_bbox_labels=y_true_bbls,
                y_pred_bbox_labels=y_pred_bbls,
                width=width,
                height=height,
                num_classes=num_classes,
                groups=grouping,
            )

        return output

    def _calculate_subgroup_metrics_fomo(
        self,
        num_classes: int,
        y_true_labels: np.ndarray,
        y_pred_labels: np.ndarray,
    ):
        """
        Calculate metrics that capture differences between subgroups of samples based on metadata,
        in a FOMO context.

        Args:
            num_classes: Number of classes
            y_true_labels: True labels
            y_pred_labels: Predicted labels
        """
        if self._no_subgroup_information():
            return None

        y_true_labels_one_hot = tf.one_hot(y_true_labels, num_classes).numpy()
        y_pred_labels_one_hot = tf.one_hot(y_pred_labels, num_classes).numpy()

        shape = TensorShape(y_pred_labels_one_hot.shape)
        weighted_xent = models.construct_weighted_xent_fn(shape, 1.0)

        def numpy_xent(y_true, y_pred_logits):
            # One item per sample
            return [
                weighted_xent(true, pred) for true, pred in zip(y_true, y_pred_logits)
            ]

        facetted_metrics = FacettedMetrics(
            per_sample_metadata=self._quantized_metadata,
            loss_fn=numpy_xent,
            stats_test_name="ttest_ind",
        )

        output = {"facetted": None, "per_key": {}}

        output["facetted"] = facetted_metrics.run_test(
            y_true=y_true_labels_one_hot,
            y_pred=y_pred_labels_one_hot,
            row_to_sample_id=self._row_to_sample_id,
        )

        for key, grouping in self._metadata_key_groupings(facetted_metrics):
            output["per_key"][key] = calculate_fomo_metrics(
                y_true_labels=y_true_labels,
                y_pred_labels=y_pred_labels,
                num_classes=num_classes,
                groups=grouping,
            )

        return output

    def _filter_and_convert_metadata(
        self,
    ):
        """
        Filter metadata to only include samples that are in our current subset of the dataset,
        and convert any numeric metadata values from strings to floats.
        """
        # Filter so we have just the samples that are listed
        specific_metadata = {
            sample_id: self._per_sample_metadata[sample_id]
            for sample_id in self._per_sample_metadata
            if sample_id in self._row_to_sample_id
        }

        # Convert any numeric metadata values from strings to floats
        for sample_id in specific_metadata:
            for key in specific_metadata[sample_id]:
                if isinstance(specific_metadata[sample_id][key], str):
                    try:
                        specific_metadata[sample_id][key] = float(
                            specific_metadata[sample_id][key]
                        )
                    except ValueError:
                        pass
        return specific_metadata

    def _prepare_metadata_for_facetted_metrics(
        self,
    ):
        """
        Prepare metadata for use in calculating subgroup metrics by filtering, converting types,
        and quantizing.
        """
        filtered_metadata = self._filter_and_convert_metadata()

        if len(filtered_metadata) == 0:
            return {}

        # Perform groupings of metadata
        return quantize_metadata(filtered_metadata)

    def _metadata_key_groupings(self, facetted_metrics: FacettedMetrics):
        """
        Generate groupings for each unique key in the metadata.

        Args:
            facetted_metrics: Instance of a FacettedMetrics object to use for groupings
        """
        # Sort so it has a stable order
        all_metadata_keys = sorted(list(set(chain(*self._per_sample_metadata.values()))))
        for key in all_metadata_keys:
            yield key, facetted_metrics._grouping_for_key(key, self._row_to_sample_id)

    def _no_subgroup_information(self):
        """
        Returns True if there is no subgroup information available based on the current metadata.
        """
        return len(self._per_sample_metadata) == 0 or len(self._row_to_sample_id) == 0
