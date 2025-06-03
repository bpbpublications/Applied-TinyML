import time
import tensorflow as tf
import numpy as np
import ei_tensorflow.inference
from .nms_layer import NMSLayer
from .output_decoder_layer import DecodeDetections
from .decode_layer import YOLODecodeLayer
from .bbox_postprocessing_layer import BBoxPostProcessingLayer
from .decode_layer_yolov4 import YOLOv4DecodeLayer

from ei_shared.types import ObjectDetectionDetails
from tensorflow.data import Dataset


def tflite_predict(
    model, dataset: Dataset, dataset_length: int, objdet_details: ObjectDetectionDetails
):
    """
    Returns TAO OD predictions for a dataset.
    """
    print("Running TAO OD inference")
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    last_log = time.time()

    pred_y = []
    for batch, _ in dataset:
        for item in batch:
            output = inference(interpreter, item, objdet_details)
            pred_y.append(output)

            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print(
                    "Profiling {0}% done".format(
                        int(100 / dataset_length * (len(pred_y) - 1))
                    ),
                    flush=True,
                )
                last_log = current_time

    print("Done inferencing")

    return np.array(pred_y, dtype=object)


def inference(interpreter, item, objdet_details: ObjectDetectionDetails):
    """
    Runs inference on a single item using a TFLite interpreter instantiated with
    a TAO model, and processes the output.
    """
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
    item_as_tensor = tf.reshape(item_as_tensor, input_shape)
    interpreter.set_tensor(input_details[0]["index"], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    output = ei_tensorflow.inference.process_output(
        output_details, output, return_np=True, remove_batch=False
    )
    if objdet_details.last_layer == "tao-yolov4":

        def process_anchors(anchors, grid_scale_xy=1.0):
            bbox_post_processing = BBoxPostProcessingLayer(grid_scale_xy=grid_scale_xy)
            bbox = bbox_post_processing(anchors)
            return bbox

        bbox_dims = 5 + objdet_details.tao_nms_attributes["numClasses"]
        bbox = process_anchors(output[:, :, -bbox_dims:])
        output = tf.concat((output[:, :, :-bbox_dims], bbox), axis=2)
        yolo_decode_layer = YOLOv4DecodeLayer()
        output = yolo_decode_layer(output)
    elif objdet_details.last_layer == "tao-yolov3":
        yolo_decode_layer = YOLODecodeLayer()
        output = yolo_decode_layer(output)

    if objdet_details.last_layer in ["tao-yolov3", "tao-yolov4"]:
        deduct_1 = False
        decode_layer = NMSLayer(
            # This `confidenceThreshold` comes from training, it isn't the same as the
            # threshold used to filter results for model testing
            # and live classification.
            output_size=objdet_details.tao_nms("keepTopK"),
            iou_threshold=objdet_details.tao_nms("iouThreshold"),
            score_threshold=objdet_details.tao_nms("scoreThreshold"),
        )
        decoded = decode_layer(output)
    elif objdet_details.last_layer in ["tao-retinanet", "tao-ssd"]:
        deduct_1 = True
        decode_layer = DecodeDetections(
            # This `confidenceThreshold` comes from training, it isn't the same as the
            # threshold used to filter results for model testing
            # and live classification.
            confidence_thresh=objdet_details.tao_nms("confidenceThreshold"),
            iou_threshold=objdet_details.tao_nms("nmsThreshold"),
            top_k=objdet_details.tao_nms("keepTopK"),
            nms_max_output_size=400,  # Use the default value
            # Set height and width to 1 to trick it into giving us relative
            # values, not pixel values
            img_height=1.0,
            img_width=1.0,
            name="decoded_predictions",
        )
        decoded = decode_layer(output)
    else:
        raise Exception(f"Unknown last layer: '{objdet_details.last_layer}'")

    # The output of DecodeDetections is [batch, top_k, 6] where the last axis is
    # [class_id, confidence, xmin, ymin, xmax, ymax]

    output = adapt_for_studio(decoded, objdet_details.tao_nms("keepTopK"), deduct_1)

    return output


def adapt_for_studio(decoded, top_k, deduct_1):
    """
    Transforms the output of the DecodeDetections layer into the format Studio expects.
    """
    # We want to turn it into a list like this for consistency with other parts of Studio
    # [([ymin, xmin, ymax, xmax], label, score)]
    assert decoded.shape[0] == 1, "Only one image per batch is expected"
    unbatched = np.reshape(decoded, (top_k, 6))
    listed = unbatched.tolist()

    def adapt_detection(item):
        label, score, xmin, ymin, xmax, ymax = item
        # Deduct 1 from the label because Studio expects labels starting from 0,
        # but the model considers 0 the background class for ssd and retinanet
        if deduct_1:
            return [[ymin, xmin, ymax, xmax], label - 1, score]
        else:
            return [[ymin, xmin, ymax, xmax], label, score]

    output = list(map(adapt_detection, listed))
    return output


def inference_and_evaluate(
    interpreter,
    item,
    objdet_details: ObjectDetectionDetails,
    specific_input_shape,
    minimum_confidence_rating: float,
    y_data,
    num_classes,
):
    """
    Runs inference on a single item using a TFLite interpreter instantiated with
    a TAO Retinanet model, and processes the output. Then evaluates the output
    against the expected output and returns the results.
    """
    raw_scores = inference(interpreter, item, objdet_details)
    # Remove scores where the 3rd element is below the minimum confidence rating
    filtered_scores = [
        score for score in raw_scores if score[2] >= minimum_confidence_rating
    ]
    height, width, _channels = specific_input_shape
    scores = ei_tensorflow.inference.compute_performance_object_detection(
        filtered_scores, width, height, y_data, num_classes
    )
    return scores
