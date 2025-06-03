from typing import Union, List

import numpy as np
import math
import json


class Centroid(object):
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def as_int(self):
        return Centroid(int(self.x), int(self.y), self.label)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return str({"x": self.x, "y": self.y, "label": self.label})


class BoundingBox(object):

    def from_x_y_h_w(x, y, h, w):
        return BoundingBox(x, y, x + w, y + h)

    def from_dict(d: dict):
        return BoundingBox(d["x0"], d["y0"], d["x1"], d["y1"])

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def close(self, other, atol) -> bool:
        return (
            np.isclose(self.x0, other.x0, atol=atol)
            and np.isclose(self.y0, other.y0, atol=atol)
            and np.isclose(self.x1, other.x1, atol=atol)
            and np.isclose(self.y1, other.y1, atol=atol)
        )

    def project(self, width: int, height: int):
        return BoundingBox(
            self.x0 * width, self.y0 * height, self.x1 * width, self.y1 * height
        )

    def floored(self):
        return BoundingBox(
            math.floor(self.x0),
            math.floor(self.y0),
            math.floor(self.x1),
            math.floor(self.y1),
        )

    def transpose_x_y(self):
        return BoundingBox(self.y0, self.x0, self.y1, self.x1)

    def clip_0_1(self):
        clip01 = lambda v: max(0, min(1, v))
        return BoundingBox(
            clip01(self.x0), clip01(self.y0), clip01(self.x1), clip01(self.y1)
        )

    def centroid(self):
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        return Centroid(cx, cy, label=None)

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

    def area(self):
        return self.width() * self.height()

    def update_with_overlap(self, other) -> bool:
        """update ourselves with any overlap. return true if there was overlap"""
        if (
            other.x0 > self.x1
            or other.x1 < self.x0
            or other.y0 > self.y1
            or other.y1 < self.y0
        ):
            return False
        if other.x0 < self.x0:
            self.x0 = other.x0
        if other.y0 < self.y0:
            self.y0 = other.y0
        if other.x1 > self.x1:
            self.x1 = other.x1
        if other.y1 > self.y1:
            self.y1 = other.y1
        return True

    def as_dict(self) -> dict:
        return {
            "x0": float(self.x0),
            "y0": float(self.y0),
            "x1": float(self.x1),
            "y1": float(self.y1),
        }

    def __eq__(self, other) -> bool:
        return self.close(other, atol=1e-8)

    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __repr__(self) -> str:
        return str(self.as_dict())

    def _intersection_area_with(self, other):
        x_a = max(self.x0, other.x0)
        x_b = min(self.x1, other.x1)
        if x_a > x_b:
            return 0
        y_a = max(self.y0, other.y0)
        y_b = min(self.y1, other.y1)
        if y_a > y_b:
            return 0
        else:
            return (x_a - x_b) * (y_a - y_b)

    def intersection_over_union(self, other):
        intersection_area = self._intersection_area_with(other)
        if intersection_area == 0:
            return 0
        union_area = self.area() + other.area() - intersection_area
        return intersection_area / union_area


class BoundingBoxLabelScore(object):

    @staticmethod
    def from_dict(d: dict):
        bbox = BoundingBox.from_dict(d["bbox"])
        return BoundingBoxLabelScore(bbox, d["label"], d["score"])

    @staticmethod
    def from_bounding_box_labels_file(fname):
        """
        Parse a bounding_box.labels file as exported from Studio and return a
        dictionary with key = source filename & value = [BoundingBoxLabelScore]
        """
        with open(fname, "r") as f:
            labels = json.loads(f.read())
            if labels["version"] != 1:
                raise Exception(f"Unsupported file version [{labels['version']}]")
            result = {}
            for fname, bboxes in labels["boundingBoxes"].items():
                bbls = []
                for bbox in bboxes:
                    x, y = bbox["x"], bbox["y"]
                    w, h = bbox["width"], bbox["height"]
                    bbls.append(
                        BoundingBoxLabelScore(
                            BoundingBox.from_x_y_h_w(x, y, h, w), bbox["label"]
                        )
                    )
                result[fname] = bbls
            return result

    @staticmethod
    def from_tf_dataset(dataset, unbatch=True):
        """returns List[List[BoundingBoxLabelScore]]"""
        dataset_bbox_labels_scores = []
        # Start from the beginning of the dataset
        dataset = dataset.take(-1)
        if unbatch:
            dataset = dataset.unbatch()
        for _image, (bboxes, labels) in dataset:
            if bboxes.shape[0] != labels.shape[0]:
                raise Exception("Mismatch in |bboxes| vs |labels|")
            bboxes = bboxes.numpy()
            instance_bbox_labels_scores = []
            for i in range(bboxes.shape[0]):
                instance_bbox_labels_scores.append(
                    BoundingBoxLabelScore(
                        bbox=BoundingBox(*bboxes[i].tolist()),
                        label=int(np.argmax(labels[i])),
                        score=None,  # ground truth
                    )
                )
            dataset_bbox_labels_scores.append(instance_bbox_labels_scores)
        return dataset_bbox_labels_scores

    @staticmethod
    def from_studio_predictions(dataset_predictions):
        """returns List[List[BoundingBoxLabelScore]]"""
        dataset_bbox_labels_scores = []
        for instance_predictions in dataset_predictions:
            instance_bbox_labels_scores = []
            for bbox, label, score in instance_predictions:
                bbox = BoundingBox(*bbox).clip_0_1()
                instance_bbox_labels_scores.append(
                    BoundingBoxLabelScore(bbox, label, score)
                )
            dataset_bbox_labels_scores.append(instance_bbox_labels_scores)
        return dataset_bbox_labels_scores

    @staticmethod
    def from_grouth_truth_samples_dict(
        samples: List[dict], img_width: int, img_height: int
    ):
        """returns List[List[BoundingBoxLabelScore]]"""
        dataset_bbox_labels_scores = []
        for sample in samples:
            bbox_labels_scores = []
            for bb in sample["boundingBoxes"]:
                # ignore entry if height or width is zero
                if bb["h"] == 0 or bb["w"] == 0:
                    continue
                bbox = (
                    BoundingBox.from_x_y_h_w(bb["y"], bb["x"], bb["w"], bb["h"])
                    .project(1.0 / img_width, 1.0 / img_height)
                    .clip_0_1()
                )
                bbox_labels_scores.append(
                    BoundingBoxLabelScore(
                        bbox=bbox,
                        label=bb["label"] - 1,  # map from 1 index to 0 index
                        score=None,  # ground truth
                    )
                )
            dataset_bbox_labels_scores.append(bbox_labels_scores)
        return dataset_bbox_labels_scores

    @staticmethod
    def from_detections_samples_dict(samples: List[dict]):
        """returns List[List[BoundingBoxLabelScore]]"""
        dataset_bbox_labels_scores = []
        for sample in samples:
            bbox_labels_scores = []
            bboxes = sample["boxes"]
            labels = sample["labels"]
            scores = sample["scores"]
            if len(set([len(bboxes), len(labels), len(scores)])) != 1:
                raise Exception(
                    "Expected 'boxes', 'labels', 'scores' to"
                    f" be the same length {sample}"
                )
            for bbox, label, score in zip(bboxes, labels, scores):
                bbox = BoundingBox(*bbox).clip_0_1()
                bbox_labels_scores.append(
                    BoundingBoxLabelScore(bbox, int(label), score)
                )
            dataset_bbox_labels_scores.append(bbox_labels_scores)
        return dataset_bbox_labels_scores

    @staticmethod
    def from_list_of_lists_of_dicts(bblss):
        """convert List[List[dict]] to List[List[BoundingBoxLabelScore]]"""
        per_image = []
        for bbls in bblss:
            per_image.append([BoundingBoxLabelScore.from_dict(e) for e in bbls])
        return per_image

    def __init__(self, bbox: BoundingBox, label: int, score: float = None):
        self.bbox = bbox
        self.label = label
        self.score = score

    def centroid(self):
        centroid = self.bbox.centroid()
        centroid.label = self.label
        return centroid

    @staticmethod
    def to_list_of_lists_of_dicts(bblss):
        """convert List[List[BoundingBoxLabelScore]] to List[List[dict]]"""
        per_image = []
        for bbls in bblss:
            per_image.append([e.as_dict() for e in bbls])
        return per_image

    def as_dict(self) -> dict:
        return {"bbox": self.bbox.as_dict(), "label": self.label, "score": self.score}

    def __eq__(self, other) -> bool:
        if self.score is None or other.score is None:
            score_equal = self.score == other.score
        else:
            score_equal = np.isclose(self.score, other.score)
        return score_equal and self.bbox == other.bbox and self.label == other.label

    def __repr__(self) -> dict:
        return str(self.as_dict())


class Labels:
    """Represents a set of labels for a classification problem"""

    def __init__(self, labels: "list[str]"):
        if len(set(labels)) < len(labels):
            raise ValueError("No duplicates allowed in label names")
        self._labels_str = labels

    # Need to upgrade to numpy >= 1.2.0 to get proper type support
    def __getitem__(self, lookup: "Union[int, np.integer, str]"):
        if isinstance(lookup, (int, np.integer)):
            if lookup < 0:
                raise IndexError(f"Index {lookup} is too low")
            if lookup >= len(self._labels_str):
                raise IndexError(f"Index {lookup} is too high")
            return Label(self, int(lookup), self._labels_str[lookup])
        elif isinstance(lookup, str):
            return Label(self, self._labels_str.index(lookup), lookup)
        else:
            raise IndexError(f"Index {lookup} is not in the list of labels")

    def __len__(self):
        return len(self._labels_str)

    def __iter__(self):
        for idx in range(0, len(self._labels_str)):
            yield Label(self, idx, self._labels_str[idx])

    def to_one_hot(self, elements: List[str]):
        if len(elements) == 0:
            raise IndexError("can't one hot an empty array")
        one_hot = np.zeros((len(elements), len(self)), dtype=int)
        for i, element in enumerate(elements):
            one_hot[i, self[element].idx] = 1
        return one_hot

    def map_to_target_indexes(self, target_labels, idxs: List[int]):
        """Map from this Labels label set to another target Labels.

        e.g.
        labels_1 = Labels(['a', 'b', 'c'])
        labels_2 = Labels(['x', 'a', 'b'])
        labels_1.map_to_target_indexes(
            labels_2, [0, 2]) == [1, None]

        since idx=0 in labels_1 is 'a', which has idx=1 in labels_2
        and idx=2 in labels_1 is 'c', which is not present in labels_2
        """

        mapped_idxs = []
        for idx in idxs:
            try:
                label_str = self[idx]._label_str
                target_idx = target_labels[label_str]
                mapped_idxs.append(target_idx._label_idx)
            except ValueError:
                mapped_idxs.append(None)
        return mapped_idxs


class Label:
    """Represents an individual label for a classification problem"""

    def __init__(self, labels: Labels, label_idx: int, label_str: str):
        self._labels = labels
        self._label_idx = label_idx
        self._label_str = label_str

    @property
    def idx(self):
        return self._label_idx

    @property
    def str(self) -> str:
        return self._label_str

    @property
    def all_labels(self):
        return self._labels

    def __eq__(self, other):
        if isinstance(other, Label):
            # Individual labels are only the same if they come from the same list
            if list(other.all_labels._labels_str) != list(self.all_labels._labels_str):
                raise ValueError("Cannot compare Label from different sets")
            return self._label_idx == other._label_idx
        raise TypeError("Cannot compare Label with non-labels")
