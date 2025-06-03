import numpy as np
import tensorflow as tf
import os, json, time, math
from typing import Optional

from tensorflow.lite.python.interpreter import Interpreter

import ei_tensorflow.utils
from ei_shared.types import ClassificationMode, ObjectDetectionDetails
from ei_tensorflow.constrained_object_detection.util import convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.util import convert_sample_bbox_and_labels_to_boundingboxlabelscores
from ei_tensorflow.constrained_object_detection.metrics import non_background_metrics
from ei_tensorflow.constrained_object_detection.metrics import match_by_near_centroids
import ei_tensorflow.tao_inference.tao_decoding
from ei_shared.metrics_utils import MetricsJson
from ei_shared.evaluator import Evaluator
from ei_shared.labels import BoundingBoxLabelScore, Labels

def ei_log(msg: str):
    print("EI_LOG_LEVEL=debug", msg, flush=True)

def infer_square_shape(x):
    flat_shape = len(x)
    height_width = int(math.sqrt(flat_shape))
    if flat_shape == height_width * height_width:
        return height_width
    raise Exception("can't derive uinflattened square shape from ", len(x))

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        # If you dont clip, casting will wrap around
        data = np.clip(np.around(data), -128, 127)
        data = data.astype(np.int8)
    if input_details[0]['dtype'] is np.uint8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.clip(np.around(data), 0, 255)
        data = data.astype(np.uint8)
    return tf.convert_to_tensor(data)

def process_output(output_details, output, return_np=False, remove_batch=True) -> 'list[float]':
    """Transforms an output tensor into a Python list, dequantizing if necessary.

    Args:
        output_details: The result of calling interpreter.get_output_details()
        data (tensor): The raw output tensor

    Returns:
        A Python list representing the output, dequantized if necessary
    """
    # If the output tensor is int8, dequantize the output data
    if output_details[0]['dtype'] is np.int8:
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        output = output.astype(np.float32)
        output = (output - zero_point) * scale
    if output_details[0]['dtype'] is np.uint8:
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        output = output.astype(np.float32)
        output = (output - zero_point) * scale

    # Most outputs have a batch dimension, which we will remove by default.
    # But some models don't (such as tao-retinanet) so we need to make it optional.
    if remove_batch and output.shape[0] == 1:
        output = output[0]

    if return_np:
        return output
    else:
        return output.tolist()

def process_output_yolov5(output_data, img_shape, version, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_data: The output of the model
        img_shape: The shape of the image, e.g. (width, height)
        version: Either 5 or 6 (for v5 or v6)
        minimum_confidence_rating: Minimum confidence rating

    Returns:
        A Python list representing the output
    """

    if (version != 5 and version != 6):
        raise Exception('process_output_yolov5 requires either version 5 or 6')

    xyxy, classes, scores = yolov5_detect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]

    rects = []
    labels = []
    score_res = []

    if (minimum_confidence_rating == None):
        minimum_confidence_rating = 0.01

    for i in range(len(scores)):
        if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
            xmin = float(xyxy[0][i])
            ymin = float(xyxy[1][i])
            xmax = float(xyxy[2][i])
            ymax = float(xyxy[3][i])

            # v5 has the absolute values (instead of 0..1)
            if (version == 5):
                xmin = xmin / img_shape[0]
                xmax = xmax / img_shape[0]
                ymin = ymin / img_shape[1]
                ymax = ymax / img_shape[1]

            # Use TensorFlow standard box format
            bbox = [ymin, xmin, ymax, xmax]

            rects.append(bbox)
            labels.append(int(classes[i]))
            score_res.append(float(scores[i]))

    raw_scores = list(zip(rects, labels, score_res))
    nms_scores = object_detection_nms(raw_scores, img_shape[0], 0.4)
    return nms_scores

def process_output_object_detection(output_details, interpreter, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_details: The result of calling interpreter.get_output_details()
        interpreter: The interpreter

    Returns:
        A Python list representing the output
    """
    # Models trained before and after our TF2.7 upgrade have different output tensor orders;
    # use names instead of indices to ensure correct functioning.
    # Create a map of name to output_details index:
    name_map = {o['name']: o['index'] for o in output_details}
    # StatefulPartitionedCall:0 is number of detections, which we can ignore
    try:
        scores = interpreter.get_tensor(name_map['StatefulPartitionedCall:1'])[0].tolist()
        labels = interpreter.get_tensor(name_map['StatefulPartitionedCall:2'])[0].tolist()
        rects = interpreter.get_tensor(name_map['StatefulPartitionedCall:3'])[0].tolist()
    except KeyError:
        # If the expected names are missing, default to legacy order
        scores = interpreter.get_tensor(output_details[2])[0].tolist()
        labels = interpreter.get_tensor(output_details[1])[0].tolist()
        rects = interpreter.get_tensor(output_details[0])[0].tolist()

    combined = list(zip(rects, labels, scores))

    # Filter out any scores that don't meet the minimum
    if minimum_confidence_rating is not None:
        return list(filter(lambda x: x[2] >= minimum_confidence_rating, combined))
    else:
        return combined

def object_detection_nms(raw_scores: list, width_height: int, iou_threshold: float=0.4):
    if len(raw_scores) == 0:
        return raw_scores

    d_boxes, d_labels, d_scores = list(zip(*raw_scores))

    n_boxes = []
    n_labels = []
    n_scores = []

    for label in np.unique(d_labels):
        mask = [ True if x == label else False for x in d_labels ]

        boxes = np.array(d_boxes)[mask]
        labels = np.array(d_labels)[mask]
        scores = np.array(d_scores)[mask]

        fixed_boxes = [box * width_height for box in boxes]

        # This takes [ymin, xmin, ymax, xmax]
        selected_indices = tf.image.non_max_suppression(
            fixed_boxes,
            scores,
            max_output_size=len(fixed_boxes),
            iou_threshold=iou_threshold,
            score_threshold=0.001).numpy()

        for ix in selected_indices:
            n_boxes.append(list(boxes[ix]))
            n_labels.append(int(labels[ix]))
            n_scores.append(float(scores[ix]))

    combined = list(zip(n_boxes, n_labels, n_scores))
    return combined

def compute_performance_object_detection(raw_detections: list, width: int, height: int,
                                         y_data: dict, num_classes: int):
    info = {
        'sampleId': y_data['sampleId'],
    }
    if len(raw_detections) > 0:
        info['boxes'], info['labels'], info['scores'] = list(zip(*raw_detections))
    else:
        info['boxes'], info['labels'], info['scores'] = [], [], []

    # If there are no ground truth bounding boxes, emit either a perfect or a zero score
    if len(y_data['boundingBoxes']) == 0:
        if len(raw_detections) == 0:
            info['mAP'] = 1
        else:
            info['mAP'] = 0
        return info

    # Our training code standardizes on the TF box format [ymin, xmin, ymax, xmax].
    # We must translate between this and what the metrics library expects.
    # TODO: Use a typed object to remove the potential for bugs.
    def convert_y_box_format(box):
        coords = ei_tensorflow.utils.convert_box_coords(box, width, height)
        # The library expects [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        return [coords[1], coords[0], coords[3], coords[2], int(box['label'] - 1), 0, 0]

    def convert_preds_format(p):
        # The library expects [xmin, ymin, xmax, ymax, class_id, confidence]
        return [p[0][1], p[0][0], p[0][3], p[0][2], int(p[1]), p[2]]

    gt = np.array(list(map(convert_y_box_format, y_data['boundingBoxes'])))
    preds = np.array(list(map(convert_preds_format, raw_detections)))

    # This is only installed on Keras container so import it only when used
    from mean_average_precision import MetricBuilder
    metric_fn_pred = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    metric_fn_pred.add(preds, gt)
    # These threshold args result in the COCO mAP
    metric_pred = metric_fn_pred.value(iou_thresholds=np.arange(0.5, 1.0, 0.05),
                                         recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
    coco_map_pred = metric_pred['mAP']

    # The mAP calculation is designed to run over an entire dataset, so it expects all classes
    # to be present. However, for a given image it is common that only some classes are present.
    # For a classifier trained on 2 classes and an image with only 1, the maximum mAP is 1 / 2.
    # For a classifier trained on 3 classes and an image with only 1, the maximum mAP is 1 / 3.
    # To make up for this, we should divide the actual mAP by the maximum mAP for that image.
    classes_in_gt = len(set([box['label'] for box in y_data['boundingBoxes']]))
    maximum_mAP = classes_in_gt / num_classes
    scaled_mAP = coco_map_pred / maximum_mAP
    info['mAP'] = float(scaled_mAP)

    return info

def run_model(mode: ClassificationMode,
              interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
              minimum_confidence_rating: Optional[float]=None, y_data=None,
              num_classes: Optional[int]=None, dir_path: Optional[str]=None,
              objdet_details: Optional[ObjectDetectionDetails]=None):
    """Runs inference with a given model and mode
    """
    if mode == 'classification' or mode == 'regression' or mode == 'anomaly-gmm' or mode == 'visual-anomaly':
        return run_vector_inference(interpreter, item, specific_input_shape)
    elif mode == 'object-detection':
        assert objdet_details is not None, 'Object detection details are required for mode object-detection'
        assert minimum_confidence_rating is not None, 'Minimum confidence rating is required for mode object-detection'
        if objdet_details.last_layer == 'mobilenet-ssd':
            return run_object_detection_inference(interpreter, item, specific_input_shape,
                                                minimum_confidence_rating, y_data, num_classes)
        elif objdet_details.last_layer == 'yolov2-akida':
            return run_akida_yolov2_inference(interpreter, item, specific_input_shape,
                                        minimum_confidence_rating, y_data, num_classes, dir_path)
        elif objdet_details.last_layer == 'yolov5':
            return run_yolov5_inference(interpreter, item, specific_input_shape, 6,
                                        minimum_confidence_rating, y_data, num_classes)
        elif objdet_details.last_layer == 'yolov5v5-drpai':
            return run_yolov5_inference(interpreter, item, specific_input_shape, 5,
                                        minimum_confidence_rating, y_data, num_classes)
        elif objdet_details.last_layer == 'yolox':
            return run_yolox_inference(interpreter, item, specific_input_shape,
                                        minimum_confidence_rating, y_data, num_classes)
        elif objdet_details.last_layer == 'yolov7':
            return run_yolov7_inference(interpreter, item, specific_input_shape,
                                        minimum_confidence_rating, y_data, num_classes)
        elif objdet_details.last_layer == 'fomo':
            return run_segmentation_inference(interpreter, item, specific_input_shape,
                                              minimum_confidence_rating, y_data)
        elif objdet_details.last_layer in ['tao-retinanet', 'tao-ssd', 'tao-yolov3', 'tao-yolov4']:
            return ei_tensorflow.tao_inference.tao_decoding.inference_and_evaluate(interpreter, item, objdet_details,
                                                                specific_input_shape, minimum_confidence_rating, y_data,
                                                                num_classes)
        elif objdet_details.last_layer == 'yolo-pro':
            return run_yolo_pro_inference(interpreter, item, specific_input_shape,
                                        minimum_confidence_rating, y_data, num_classes)
        else:
            raise ValueError('Invalid object detection last layer "' + str(objdet_details.last_layer) + '"')
    else:
        raise ValueError('Invalid mode "' + mode + '"')

def invoke(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]'):
    """Invokes the Python TF Lite interpreter with a given input
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)

    # now that we reshape below here, I think this can go...
    if specific_input_shape is not None:
        if (np.prod(item_as_tensor.shape) != np.prod(specific_input_shape)):
            raise Exception('Invalid number of features, expected ' + str(np.prod(specific_input_shape)) + ', but got ' +
                str(np.prod(item_as_tensor.shape)) + ' (trying to reshape into ' + json.dumps(np.array(specific_input_shape).tolist()) + '). ' +
                'Try re-generating features and re-training your model.')

        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)

    # check input shape of the model and reshape to that (e.g. adds batch dim already)
    if (np.prod(item_as_tensor.shape) != np.prod(input_details[0]['shape'])):
        raise Exception('Invalid number of features, expected ' + str(np.prod(input_details[0]['shape'])) + ', but got ' +
            str(np.prod(item_as_tensor.shape)) + ' (trying to reshape into ' + json.dumps(np.array(input_details[0]['shape']).tolist()) + '). ' +
            'Try re-generating features and re-training your model.')
    item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])

    # invoke model
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = process_output(output_details, output)
    output = np.array(output)
    return output, output_details

def run_vector_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]'):
    """Runs inference that produces a vector output (classification or regression)
    """
    output, output_details = invoke(interpreter, item, specific_input_shape)
    return output

def run_object_detection_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                                   minimum_confidence_rating: float, y_data, num_classes):
    """Runs inference that produces an object detection output
    """
    height, width, _channels = specific_input_shape

    output, output_details = invoke(interpreter, item, specific_input_shape)
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not num_classes:
        raise ValueError('num_classes must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')
    raw_detections = process_output_object_detection(output_details, interpreter, minimum_confidence_rating)
    scores = compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)
    return scores

def run_segmentation_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                               minimum_confidence_rating: float, y_data: list):
    """Runs inference that produces an object detection output
    """

    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape
    if width != height:
        raise Exception(f"Only square input is supported; not {specific_input_shape}")
    input_width_height = width

    output, output_details = invoke(interpreter, item, specific_input_shape)

    _batch, width, height, num_classes_including_background = output_details[0]['shape']
    if width != height:
        raise Exception(f"Only square output is supported; not {output_details[0]['shape']}")
    output_width_height = width

    # convert y_true to list of BoundingBoxLabelScores. note: this data is
    # 1 indexed already so covers the class=0 for implicit background
    y_true_boxes_labels_scores = convert_sample_bbox_and_labels_to_boundingboxlabelscores(
        y_data['boundingBoxes'], input_width_height)

    # convert model output to list of BoundingBoxLabelScores including fusing
    # of adjacent boxes. retains class=0 from segmentation output.
    y_pred_boxes_labels_scores = convert_segmentation_map_to_object_detection_prediction(
        output, minimum_confidence_rating, fuse=True)

    # do alignment by centroids
    y_true_labels, y_pred_labels, debug_info = match_by_near_centroids(
        y_true_boxes_labels_scores, y_pred_boxes_labels_scores,
        output_width_height=output_width_height,
        min_normalised_distance=0.2,
        return_debug_info=True)

    _precision, _recall, f1 = non_background_metrics(
        y_true_labels, y_pred_labels,
        num_classes_including_background)

    # prepare debug data
    debug_data = {}
    for key in ['y_trues', 'y_preds', 'assignments', 'normalised_min_distance', 'all_pairwise_distances', 'unassigned_y_true_idxs', 'unassigned_y_pred_idxs']:
        if key in debug_info:
            # the centroid objects are not JSON serializable
            if key == 'y_trues' or key == 'y_preds':
                debug_data[key] = [{"x": bls.x, "y": bls.y, "label": bls.label} for bls in debug_info[key]]
            elif key == 'assignments':
                debug_data[key] = [{"yp": a.yp, "yt": a.yt, "label": a.label, "distance": a.distance} for a in debug_info[key]]
            else:
                debug_data[key] = debug_info[key]

    debug_info_json = json.dumps(debug_data)

    # package up into info dict
    # as final step to return to studio map labels by -1 to remove class=0
    # background class.
    boxes = [list(bls.bbox) for bls in y_pred_boxes_labels_scores]
    labels = [bls.label-1 for bls in y_pred_boxes_labels_scores]
    scores = [bls.score for bls in y_pred_boxes_labels_scores]

    # note that mAP is set to f1 even though it is not actually a mAP score,
    # however has been use as mAP in studio until a separate f1 score was introduced.
    # we continue to set the value for mAP to not break the API.
    return {
        'sampleId': y_data['sampleId'],
        'boxes': boxes, 'labels': labels, 'scores': scores,
        'f1': f1,
        'mAP': f1,
        'precision': _precision,
        'recall': _recall,
        'debugInfoJson': debug_info_json, # used for toggling bounding boxes in live classification
        'y_true_labels': y_true_labels,
        'y_pred_labels': y_pred_labels
    }

def yolov5_class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def yolov5_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = yolov5_class_filter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def run_akida_yolov2_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                          minimum_confidence_rating: float, y_data: list, num_classes: int, output_directory: str):
    import pickle
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape

    with open(os.path.join(output_directory, "akida_yolov2_anchors.pkl"), 'rb') as handle:
        anchors = pickle.load(handle)

    output, output_details = invoke(interpreter, item, specific_input_shape)
    h, w, c = output.shape
    output = output.reshape((h, w, len(anchors), 4 + 1 + num_classes))

    raw_detections = ei_tensorflow.brainchip.model.process_output_yolov2(output, (width, height), num_classes, anchors)
    return compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)

def run_yolov5_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                          version: int, minimum_confidence_rating: float, y_data: list, num_classes):
    """Runs inference that produces an object detection output
    """
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape

    output, output_details = invoke(interpreter, item, specific_input_shape)
    # expects to have batch dim here
    output = np.expand_dims(output, axis=0)

    raw_detections = process_output_yolov5(output, (width, height),
        version, minimum_confidence_rating)
    return compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)

def run_yolox_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                        minimum_confidence_rating: float, y_data: list, num_classes):
    """Runs inference that produces an object detection output
    """
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape

    output, output_details = invoke(interpreter, item, specific_input_shape)
    # expects to have batch dim here
    output = np.expand_dims(output, axis=0)

    raw_detections = process_output_yolox(output, img_size=width, minimum_confidence_rating=minimum_confidence_rating)
    return compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)

def run_yolo_pro_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                        minimum_confidence_rating: float, y_data: list, num_classes):
    """Runs inference that produces an object detection output
    """
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape

    output, output_details = invoke(interpreter, item, specific_input_shape)
    # expects to have batch dim here
    output = np.expand_dims(output, axis=0)

    raw_detections = process_output_yolo_pro(output, img_size=width, minimum_confidence_rating=minimum_confidence_rating)
    return compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)

def yolox_detect(output, img_size):

    def yolox_postprocess(outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def yolox_interpret(boxes, scores, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        dets = np.concatenate(
            [valid_boxes[:], valid_scores[:, None], valid_cls_inds[:, None]], 1
        )

        return dets

    predictions = yolox_postprocess(output, tuple([ img_size, img_size ]))[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    dets = yolox_interpret(boxes_xyxy, scores, score_thr=0.01)

    xyxy = [
        [], [], [], []
    ]
    classes = []
    scores = []

    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    for i in range(0, len(final_boxes)):
        box = final_boxes[i]

        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        xyxy[0].append(xmin / img_size)
        xyxy[1].append(ymin / img_size)
        xyxy[2].append(xmax / img_size)
        xyxy[3].append(ymax / img_size)

        classes.append(final_cls_inds[i])

        scores.append(final_scores[i])

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


def process_output_yolox(output_data, img_size, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_data: The output of the model
        img_size: The width of the image. Image is expected to be square, e.g. (width, width)
        minimum_confidence_rating: Minimum confidence rating

    Returns:
        A Python list representing the output
    """

    xyxy, classes, scores = yolox_detect(output_data, img_size) #boxes(x,y,x,y), classes(int), scores(float) [25200]

    rects = []
    labels = []
    score_res = []

    if (minimum_confidence_rating == None):
        minimum_confidence_rating = 0.01

    for i in range(len(scores)):
        if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
            xmin = float(xyxy[0][i])
            ymin = float(xyxy[1][i])
            xmax = float(xyxy[2][i])
            ymax = float(xyxy[3][i])

            # Use TensorFlow standard box format
            bbox = [ymin, xmin, ymax, xmax]

            rects.append(bbox)
            labels.append(int(classes[i]))
            score_res.append(float(scores[i]))

    raw_scores = list(zip(rects, labels, score_res))
    nms_scores = object_detection_nms(raw_scores, img_size, 0.4)
    return nms_scores

# this is the top k score selection taken from
# https://github.com/edgeimpulse/yolo-pro/blob/87be31bcdca9954a5586671f95301cbe8ed372c5/ei_yolo/decoding.py#L167
def _top_k_detection(boxes, scores, threshold, max_detections):
    """ single instance top_k extraction """

    assert len(boxes.shape) == 2
    assert len(scores.shape) == 2
    num_anchors, coords = boxes.shape
    assert coords == 4  # xyxy
    assert scores.shape[0] == num_anchors

    boxes = np.array(boxes)
    scores = np.array(scores)

    patch_idx, classes = np.where(scores > threshold)

    num_detections_above_threshold = len(patch_idx)
    if num_detections_above_threshold == 0:
        return { 'num_detections': 0 }

    top_k_sorted_args = np.argsort(-scores[patch_idx, classes])[:max_detections]

    num_detections = len(top_k_sorted_args)
    result = { 'num_detections': num_detections}

    top_boxes = []
    top_confidence = []
    top_classes = []
    for idx in top_k_sorted_args:
        pi = patch_idx[idx]
        ci = classes[idx]
        top_boxes.append(boxes[pi])
        top_confidence.append(scores[pi,ci])
        top_classes.append(classes[idx])

    result['boxes'] = np.stack(top_boxes)
    result['confidence'] = np.array(top_confidence)
    result['classes'] = np.array(top_classes)

    return result


def top_k_results(boxes, scores, threshold, max_detections_per_image):
    """ batched top_k extraction, with -1 fill as keras_cv.NonMaxSuppression """
    num_batches = len(boxes)
    fill_value = -1
    batch_results = {
        'num_detections': np.ones((num_batches,), dtype=int) * fill_value,
        'boxes': np.ones((num_batches, max_detections_per_image, 4), dtype=float) * fill_value,
        'confidence': np.ones((num_batches, max_detections_per_image), dtype=float) * fill_value,
        'classes': np.ones((num_batches, max_detections_per_image), dtype=int) * fill_value,
    }
    for i in range(num_batches):
        result = _top_k_detection(boxes[i], scores[i],
                                  threshold=threshold, max_detections=max_detections_per_image)
        num_detections = result['num_detections']
        batch_results['num_detections'][i] = num_detections
        if num_detections == 0: continue
        batch_results['boxes'][i,:num_detections] = result['boxes']
        batch_results['confidence'][i,:num_detections] = result['confidence']
        batch_results['classes'][i,:num_detections] = result['classes']
    return batch_results

def yolo_pro_adapt_for_studio(decoded):
    # We want to turn it into a list as follows for consistency with other parts of Studio
    # [([ymin, xmin, ymax, xmax], label, score)]
    assert ((decoded['boxes'].shape[0] == decoded['classes'].shape[0]) and (decoded['classes'].shape[0] == decoded['confidence'].shape[0]) and (decoded['confidence'].shape[0] == 1)), "Only one image per batch is expected"

    boxes = np.squeeze(decoded['boxes'], axis=0)
    xmin, ymin, xmax, ymax = zip(*boxes)
    conf = np.squeeze(decoded['confidence'], axis=0)
    cls = np.squeeze(decoded['classes'], axis=0)

    listed = []
    for idx, _ in enumerate(conf):
        ##if (conf[idx] < 0) or (cls[idx] < 0) or (ymin[idx] < 0) or (xmin[idx] < 0) or (ymax[idx] < 0) or (xmax[idx] < 0):
        ##    listed.append([0]*6)
        ##else:
        ##    listed.append([xmin[idx], ymin[idx], xmax[idx], ymax[idx], cls[idx], conf[idx]])
        listed.append([xmin[idx], ymin[idx], xmax[idx], ymax[idx], cls[idx], conf[idx]])

    def adapt_detection(item):
        xmin, ymin, xmax, ymax, label, score  = item
        return [[ymin, xmin, ymax, xmax], label, score]

    output = list(map(adapt_detection, listed))
    return output

def process_output_yolo_pro(output_data, img_size, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_data: The output of the model
        img_size: The width of the image. Image is expected to be square, e.g. (width, width)
        minimum_confidence_rating: Minimum confidence rating

    Returns:
        A Python list representing the output
    """

    if (minimum_confidence_rating == None):
        minimum_confidence_rating = 0.01

    boxes = output_data[..., :4]
    scores = output_data[..., 4:]

    decoded_y_pred = top_k_results(boxes, scores, threshold=minimum_confidence_rating, max_detections_per_image=1000)
    raw_scores = yolo_pro_adapt_for_studio(decoded_y_pred)
    nms_scores = object_detection_nms(raw_scores, img_size, 0.4)
    return nms_scores

def run_yolov7_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                         minimum_confidence_rating: float, y_data: list, num_classes):
    """Runs inference that produces an object detection output
    """
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    height, width, _channels = specific_input_shape
    if width != height:
        raise Exception(f"Only square input is supported; not {specific_input_shape}")
    input_width_height = width

    output, output_details = invoke(interpreter, item, specific_input_shape)

    raw_detections = process_output_yolov7(output, width=width, height=height, minimum_confidence_rating=minimum_confidence_rating)
    return compute_performance_object_detection(raw_detections, width, height, y_data, num_classes)


def process_output_yolov7(output_data, width, height, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_details: The result of calling interpreter.get_output_details()
        interpreter: The interpreter

    Returns:
        A Python list representing the output
    """

    rects = []
    labels = []
    score_res = []

    if (minimum_confidence_rating == None):
        minimum_confidence_rating = 0.01

    for i, (batch_id, xmin, ymin, xmax, ymax, cls_id, score) in enumerate(output_data):
        # values are absolute, map back to 0..1
        xmin = xmin / width
        xmax = xmax / width
        ymin = ymin / height
        ymax = ymax / height

        # Use TensorFlow standard box format
        bbox = [ymin, xmin, ymax, xmax]

        rects.append(bbox)
        labels.append(int(cls_id))
        score_res.append(score)

    raw_scores = list(zip(rects, labels, score_res))
    return raw_scores

def prepare_interpreter(dir_path, model_path, num_threads=None):
    """Instantiates an interpreter, allocates its tensors, and returns it."""
    lite_file_path = os.path.join(dir_path, os.path.basename(model_path))
    if isinstance(num_threads, int):
        print(f'Using {num_threads} threads for inference.', flush=True)
    else:
        num_threads = None
    interpreter = tf.lite.Interpreter(model_path=lite_file_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

def map_test_label_to_train(test_ix, train_labels, test_labels, zero_index=True):
    """Converts a test label index to an index relevant to the original set of training labels"""

    # For FOMO we work with 1-indexed labels, but the original label map is always 0-indexed
    adjust_index = 0 if zero_index else 1

    actual_label = test_labels[test_ix - adjust_index]

    # Test label not in training labels? Use an out-of-range index.
    # These label indices are only used for profiling results in Python.
    # The studio only sees the original set of labels, so any index not present in the training set
    # is fine here.
    if actual_label not in train_labels:
        return -1
    return train_labels.index(actual_label) + adjust_index

def classify_keras(input_x_file, input_y_file, mode: ClassificationMode, output_file, dir_path,
                   model_path, model_head_path, specific_input_shape, use_tflite, layer_input_name,
                   layer_output_name, class_names_training, class_names_testing,
                   minimum_confidence_rating,
                   model_variant,
                   metrics_fname_prefix,
                   objdet_details: Optional[ObjectDetectionDetails]=None,
                   per_sample_metadata: Optional[dict]=None,
                   predictions_path: Optional[str]=None):
    y_true = None
    num_classes = len(class_names_training)

    input = np.load(input_x_file, mmap_mode='r')
    if (not isinstance(input[0], (np.ndarray))):
        input = np.array([ input ])

    if mode == 'object-detection':
        y_true = ei_tensorflow.utils.load_y_structured('/', input_y_file, len(input))
        sample_ids = []
        for row in y_true:
            for box in row['boundingBoxes']:
                # Studio is passing label indices using the testing dataset
                # This is ok for other models as we don't profile in Python; we just feed-back
                # the raw results and map to the correct labels later.
                # For structured data we profile in Python so we need the correct labels.
                box['label'] = map_test_label_to_train(box['label'],
                    class_names_training, class_names_testing, False)
            sample_ids.append(row['sampleId'])
    elif mode == 'visual-anomaly':
        try:
            y_true = np.load(input_y_file, mmap_mode='r')
            y_true = y_true[:,0]
        # If combined with object detection, the input_y_file file would be
        # structured. This should be improved to allow testing of both models
        # simultaneously
        except Exception as e:
            y_true = None
    elif mode == 'regression':
        y_true_info = np.load(input_y_file, mmap_mode='r')
        y_true_idxs = y_true_info[:, 0]  # first col is 1 based index in class_names_testing
        sample_ids = y_true_info[:,1] # this is the sample IDs
        y_true = np.array([float(class_names_testing[i-1]) for i in y_true_idxs])
    elif mode == 'classification':
        y_true_info = np.load(input_y_file, mmap_mode='r')
        sample_ids = y_true_info[:,1] # this is the sample IDs
        y_true = y_true_info[:,0]  # just strip first column
    else:
        raise Exception(f"Unsupport mode [{mode}]")

    if predictions_path is not None:
        if mode == 'object-detection':
            with open(predictions_path, 'r') as f:
                pred_y = json.loads(f.read())
                pred_y = map_object_detection_bboxes(pred_y, y_true, minimum_confidence_rating, specific_input_shape, num_classes)
        else:
            pred_y = np.load(predictions_path)

    else:
        # Predictions array
        pred_y = []

        # Make sure we log every 10 seconds
        LOG_MIN_INTERVAL_S = 10
        last_log_time = time.time()
        showed_slow_warning = False

        # In this code path, we use a TensorFlow Lite model
        if use_tflite:
            interpreter = prepare_interpreter(dir_path, model_path)

            if model_head_path:
                interpreter_head = prepare_interpreter(dir_path, model_head_path)
                scorer_input_details = interpreter_head.get_input_details()
                scorer_shape = scorer_input_details[0]['shape'][1:]

            for i, item in enumerate(input):
                if model_head_path:
                    features = run_model(mode=mode,
                                        interpreter=interpreter,
                                        item=item,
                                        specific_input_shape=specific_input_shape,
                                        minimum_confidence_rating=minimum_confidence_rating,
                                        y_data=y_true[i] if mode == 'object-detection' else None,
                                        num_classes=num_classes,
                                        dir_path=dir_path,
                                        objdet_details=objdet_details)

                    features = features.astype(np.float32)
                    single_y_pred = run_model(mode=mode,
                                        interpreter=interpreter_head,
                                        item=features,
                                        specific_input_shape=scorer_shape,
                                        minimum_confidence_rating=minimum_confidence_rating,
                                        y_data=None,
                                        num_classes=num_classes,
                                        dir_path=dir_path,
                                        objdet_details=objdet_details)
                else:
                    single_y_pred = run_model(mode=mode,
                                    interpreter=interpreter,
                                    item=item,
                                    specific_input_shape=specific_input_shape,
                                    minimum_confidence_rating=minimum_confidence_rating,
                                    y_data=y_true[i] if mode == 'object-detection' else None,
                                    num_classes=num_classes,
                                    dir_path=dir_path,
                                    objdet_details=objdet_details)

                pred_y.append(single_y_pred)

                # Log a message if enough time has elapsed
                current_time = time.time()
                if last_log_time + LOG_MIN_INTERVAL_S < current_time:
                    message = '{0}% done'.format(int(100 / len(input) * i))
                    if not showed_slow_warning:
                        message += ' (this can take a while for large datasets)'
                        showed_slow_warning = True
                    print(message, flush=True)
                    last_log_time = current_time

        # Otherwise, we can expect a 1.13 .pb file, and we need to do more complex things
        # to use it.
        else:
            # See https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(graph_def, inputs, outputs):
                def _imports_graph_def():
                    tf.compat.v1.import_graph_def(graph_def, name="")
                wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
                import_graph = wrapped_import.graph
                return wrapped_import.prune(
                    tf.nest.map_structure(import_graph.as_graph_element, inputs),
                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

            graph_def = tf.compat.v1.GraphDef()
            pb_file_path = os.path.join(dir_path, 'trained.pb')
            graph_def.ParseFromString(open(pb_file_path, 'rb').read())
            model_func = wrap_frozen_graph(
                graph_def, inputs=layer_input_name,
                outputs=layer_output_name)

            for item in input:
                item_as_tensor = tf.convert_to_tensor(item)
                item_as_tensor = tf.expand_dims(item_as_tensor, 0)
                output = model_func(item_as_tensor)
                scores = output[0].numpy().tolist()
                pred_y.append(scores)

    # calculate mode specific metrics
    metrics = None
    try:
        if mode == 'regression':
            evaluator = Evaluator(per_sample_metadata, sample_ids)
            eval_result = evaluator.regression(
                y_true=y_true,
                y_pred=pred_y,
            )
            metrics = eval_result.metrics

        elif mode == 'classification':
            # map the y_true idxs ( in testing label set ) to the corresponding
            # indexes in the training label set.
            training_labels = Labels(class_names_training)
            test_labels = Labels(class_names_testing)
            y_true_in_training_idxs = test_labels.map_to_target_indexes(
                target_labels = training_labels,
                idxs = y_true-1)  # note: y_true is 1 idxed at this point

            # at this point examples from y_true ( test ) may include labels that
            # are in test, but not in training. these are marked as None
            # and we explicitly filter them to be consistent with the "Model
            # testing results" confusion matrix which doesn't include them.
            filtered_y_true_idxs = []
            filtered_y_pred_probs = []
            for idx, y_pred in zip(y_true_in_training_idxs, pred_y):
                if idx is not None:
                    filtered_y_true_idxs.append(idx)
                    filtered_y_pred_probs.append(y_pred)

            # finally convert indexes to one hot values, and stack predictions
            # into single array for metrics calculation.
            y_true_one_hot = training_labels.to_one_hot(filtered_y_true_idxs)
            filtered_y_pred_probs = np.stack(filtered_y_pred_probs)
            evaluator = Evaluator(per_sample_metadata, sample_ids)
            eval_result = evaluator.classification(y_true_one_hot=y_true_one_hot,
                                                        y_pred_probs=filtered_y_pred_probs,
                                                        class_names=class_names_training)
            metrics = eval_result.metrics

        elif mode == 'object-detection':
            if objdet_details is None:
                raise ValueError('Object detection details are required for mode object-detection')
            if objdet_details.last_layer == 'fomo':
                evaluator = Evaluator(per_sample_metadata, sample_ids)
                eval_result = evaluator.fomo(
                    class_names=class_names_training,
                    y_true_labels=np.array([pred["y_true_labels"] for pred in pred_y]),
                    y_pred_labels=np.array([pred['y_pred_labels'] for pred in pred_y]),
                )
                metrics = eval_result.metrics
            else:
                # late import this since 'pycocotools' not installed on all
                # containers importing ei_tensorflow.inference (?)
                width, height, _num_channels = specific_input_shape
                y_true_bbls = BoundingBoxLabelScore.from_grouth_truth_samples_dict(
                    y_true, width, height)
                y_pred_bbls = BoundingBoxLabelScore.from_detections_samples_dict(pred_y)

                evaluator = Evaluator(per_sample_metadata, sample_ids)
                eval_result = evaluator.object_detection(
                    class_names=class_names_training,
                    width=width,
                    height=height,
                    y_true_bbls=y_true_bbls,
                    y_pred_bbls=y_pred_bbls,
                )
                metrics = eval_result.metrics
                metrics['class_names'] = class_names_training

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Failed to calculate metrics [{e}]", flush=True)

    if metrics is not None:
        # merge with existing validation metrics
        metrics_json = MetricsJson(mode=mode, filename_prefix=metrics_fname_prefix)
        metrics_json.set('test', model_variant, metrics)

    # convert from the raw y_pred value to the numpyed list expected
    # ( if we are in the use_tflite flow)
    if use_tflite:
        pred_y = [ np.array(v).tolist() for v in pred_y ]

    # round to max. 5 digits behind the . to save on space (classification only)
    if mode == 'classification':
        pred_y_rounded = [[round(val, 5) if isinstance(val, (int, float)) else val for val in row] for row in pred_y]
    else:
        pred_y_rounded = pred_y

    if output_file:
        with open(output_file, 'w') as f:
            f.write(json.dumps(pred_y_rounded, separators=(',', ':')))
    else:
        print('Begin output')
        print(json.dumps(pred_y_rounded, separators=(',', ':')))
        print('End output')

def map_object_detection_bboxes(
    predictions,
    y_true,
    minimum_confidence_rating,
    input_shape,
    num_classes
):
    #TODO, we can do custom post-processing here
    width, height, _num_channels = input_shape

    if minimum_confidence_rating is None:
        minimum_confidence_rating = 0

    new_predictions = []

    # Calculate scores for each sample
    for ix in range(0, len(predictions)):
        scores = map_object_detection_bbox(predictions[ix], y_true[ix], minimum_confidence_rating, width, height, num_classes)
        new_predictions.append(scores)

    return new_predictions

def map_object_detection_bbox(
    y_prediction,
    y_true,
    minimum_confidence_rating,
    width,
    height,
    num_classes
):
    # Filter out boxes with a score below the confidence rating
    sample_boxes = [box for box in y_prediction if box[2] >= minimum_confidence_rating]
    # Compute performance, e.g. mAP
    scores = compute_performance_object_detection(sample_boxes, width, height, y_true, num_classes)
    return scores
