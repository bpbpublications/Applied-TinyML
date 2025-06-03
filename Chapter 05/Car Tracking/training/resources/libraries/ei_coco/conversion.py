from typing import List

from ei_shared.labels import BoundingBoxLabelScore

def convert_y_true_boundingbox_label_scores_to_coco_groundtruth(
    y_true_bbox_labels: List[List[BoundingBoxLabelScore]],
    img_width: int,
    img_height: int,
    num_classes: int,
):

    images = [{"id": str(i)} for i in range(len(y_true_bbox_labels))]

    categories = [{"id": i} for i in range(num_classes)]

    annotations = []
    annotation_idx = 1
    for img_id, bbox_label_scores in enumerate(y_true_bbox_labels):
        for bbox_label_score in bbox_label_scores:
            # project bbox from normalised coords to pixel space width/height
            bbox_ps = bbox_label_score.bbox.project(img_width, img_height).floored()
            annotations.append(
                {
                    "image_id": str(img_id),
                    "iscrowd": 0,
                    "category_id": int(bbox_label_score.label),
                    "bbox": [bbox_ps.x0, bbox_ps.y0, bbox_ps.width(),
                             bbox_ps.height()],
                    "area": bbox_ps.area(),
                    "id": annotation_idx,
                }
            )
            annotation_idx += 1

    return {"images": images, "categories": categories, "annotations": annotations}

def convert_y_pred_boundingbox_label_scores_to_coco_detections(
    y_pred_bbox_labels: List[List[BoundingBoxLabelScore]],
    img_width: int,
    img_height: int,
):
    coco_predictions = []
    detection_idx = 1

    for img_id, img_detections in enumerate(y_pred_bbox_labels):
        for bbox_label_score in img_detections:
            # project bbox from normalised coords to pixel space width/height
            bbox_ps = bbox_label_score.bbox.project(img_width, img_height).floored()
            coco_predictions.append(
                {
                    "image_id": str(img_id),
                    "category_id": int(bbox_label_score.label),
                    "bbox": [bbox_ps.x0, bbox_ps.y0, bbox_ps.width(),
                             bbox_ps.height()],
                    "score": bbox_label_score.score,
                    "id": detection_idx,
                }
            )
            detection_idx += 1

    return coco_predictions


