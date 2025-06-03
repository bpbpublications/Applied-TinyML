from __future__ import print_function
import json
import time
import traceback
import os
import numpy as np
import tensorflow as tf
import json, datetime, time, traceback
import shutil, operator, functools, time, subprocess, math
from typing import Optional, List, Dict, Callable, Literal
import ei_tensorflow.inference
import ei_tensorflow.brainchip.model
from concurrent.futures import ThreadPoolExecutor
import ei_tensorflow.utils
from ei_shared.types import ClassificationMode, ObjectDetectionDetails, CustomModelVariantInfo
import ei_tensorflow.tao_inference.tao_decoding
from ei_shared.labels import BoundingBoxLabelScore

from ei_shared.metrics_utils import MetricsJson, sanitize_for_json

from ei_shared.evaluator import Evaluator, EvalResult

from ei_tensorflow.constrained_object_detection.util import (
    batch_convert_segmentation_map_to_object_detection_prediction,
)
from ei_tensorflow.constrained_object_detection.metrics import (
    dataset_match_by_near_centroids,
)
from .perf_profiling import check_if_model_runs_on_mcu

def ei_log(msg: str):
    print("EI_LOG_LEVEL=debug", msg, flush=True)

def tflite_predict(model, validation_dataset, dataset_length, item_feature_axes: Optional[list]=None):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        if item_feature_axes:
            item_as_tensor = np.take(item_as_tensor, item_feature_axes)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        scores = ei_tensorflow.inference.process_output(output_details, output)
        pred_y.append(scores)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def tflite_predict_object_detection(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            interpreter.invoke()
            rect_label_scores = ei_tensorflow.inference.process_output_object_detection(output_details, interpreter)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

# Y_test is required to generate anchors for YOLOv2 output decoding
def tflite_predict_yolov2(model, validation_dataset, Y_test, dataset_length, num_classes, output_directory):
    import pickle

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(os.path.join(output_directory, "akida_yolov2_anchors.pkl"), 'rb') as handle:
        anchors = pickle.load(handle)

    last_log = time.time()
    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            h, w, c = output.shape
            output = output.reshape((h, w, len(anchors), 4 + 1 + num_classes))
            rect_label_scores = ei_tensorflow.brainchip.model.process_output_yolov2(output, (width, height), num_classes, anchors)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    result = np.array(pred_y, dtype=object)
    return result

def tflite_predict_yolov5(model, version, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = np.array(ei_tensorflow.inference.process_output(output_details, output))
            # expects to have batch dim here, eg (1, 5376, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolov5(output, (width, height),
                version)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolox(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)

            _batch, width, height, _channels = input_details[0]['shape']
            if width != height:
                raise Exception(f"expected square input, got {input_details[0]['shape']}")

            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = np.array(ei_tensorflow.inference.process_output(output_details, output))
            # expects to have batch dim here, eg (1, 5376, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolox(output, img_size=width)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolov7(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = ei_tensorflow.inference.process_output(output_details, output)
            rect_label_scores = ei_tensorflow.inference.process_output_yolov7(output,
                width=input_details[0]['shape'][1], height=input_details[0]['shape'][2])
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_segmentation(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    last_log = time.time()

    y_pred = []
    for item, _ in validation_dataset.take(-1):
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = ei_tensorflow.inference.process_output(output_details, output)
        y_pred.append(output)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(y_pred) - 1))), flush=True)
            last_log = current_time

    y_pred = np.stack(y_pred)

    return y_pred

def tflite_predict_yolo_pro(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = np.array(ei_tensorflow.inference.process_output(output_details, output))
            # expects to have batch dim here, eg (1, 2100, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolo_pro(output, img_size=width)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def get_tensor_details(tensor):
    """Obtains the quantization parameters for a given tensor"""
    details = {
        'dataType': None,
        'name': tensor['name'],
        'shape': tensor['shape'].tolist(),
        'quantizationScale': None,
        'quantizationZeroPoint': None
    }
    if tensor['dtype'] is np.int8:
        details['dataType'] = 'int8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.uint8:
        details['dataType'] = 'uint8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.float32:
        details['dataType'] = 'float32'
    else:
        raise Exception('Model tensor has an unknown datatype, ', tensor['dtype'])

    return details

def get_io_details(model, model_type):
    """Gets the input and output datatype and quantization details for a model"""
    interpreter = tf.lite.Interpreter(model_content=model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = list(map(get_tensor_details, input_details))
    outputs = list(map(get_tensor_details, output_details))

    return {
        'modelType': model_type,
        'inputs': inputs,
        'outputs': outputs
    }

def make_predictions_tflite(mode: ClassificationMode, model, x_dataset, y, num_classes,
                            output_directory, item_feature_axes: Optional[list]=None,
                            objdet_details: Optional[ObjectDetectionDetails]=None):
    if mode == 'object-detection':
        if objdet_details is None:
            raise ValueError('objdet_details must be provided for object-detection mode')
        if objdet_details.last_layer == 'mobilenet-ssd':
            return tflite_predict_object_detection(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov5':
            return tflite_predict_yolov5(model, 6, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov2-akida':
            return tflite_predict_yolov2(model, x_dataset, y, len(y), num_classes, output_directory)
        elif objdet_details.last_layer == 'yolov5v5-drpai':
            return tflite_predict_yolov5(model, 5, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolox':
            return tflite_predict_yolox(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov7':
            return tflite_predict_yolov7(model, x_dataset, len(y))
        elif objdet_details.last_layer in ['tao-retinanet', 'tao-ssd', 'tao-yolov3', 'tao-yolov4']:
            return ei_tensorflow.tao_inference.tao_decoding.tflite_predict(model, x_dataset, len(y), objdet_details)
        elif objdet_details.last_layer == 'fomo':
            return tflite_predict_segmentation(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolo-pro':
            return tflite_predict_yolo_pro(model, x_dataset, len(y))
        else:
            raise Exception(f'Expecting a supported object detection last layer (got {objdet_details.last_layer})')
    elif mode == 'visual-anomaly':
        raise Exception('Expecting a supported mode to make predictions (visual-anomaly is not)')
    else:
        return tflite_predict(model, x_dataset, len(y), item_feature_axes)

def profile_model(model_type: Literal['float32', 'int8', 'akida'],
                  model: Optional[bytes],
                  model_file: Optional[str],
                  akida_model_path: Optional[str],
                  validation_dataset: tf.data.Dataset,
                  Y_test: np.ndarray,
                  X_samples: Optional[np.ndarray],
                  Y_samples: Optional[np.ndarray],
                  has_samples: bool,
                  memory: Optional[Dict],
                  mode: ClassificationMode,
                  class_names: List[str],
                  item_feature_axes: Optional[list],
                  per_sample_metadata: Optional[dict],
                  # A dictionary containing the sample IDs in row order for each split (training and validation)
                  sample_id_details: Optional["dict[str, list[int]]"],
                  objdet_details: Optional[ObjectDetectionDetails],
                  custom_model_variant: Optional[CustomModelVariantInfo] = None):
    """Calculates performance statistics for a model, including both metrics and memory usage"""

    num_classes = len(class_names)
    is_custom_variant = False

    # The result of evaluating a model. We instantiate an empty one here to provide default values.
    eval_result = EvalResult()
    # A sample of predictions that is used for the feature explorer
    prediction_samples = []

    if custom_model_variant is not None:
        is_custom_variant = True
    elif model_file is not None:
        model_path = model_file
    elif akida_model_path is not None:
        model_path = akida_model_path
    else:
        raise ValueError('Expecting either a model file or an Akida model path')

    if mode != 'visual-anomaly':
        if is_custom_variant:
            # Load predictions from file
            if mode == 'object-detection':
                with open(custom_model_variant.outputPredictionsPath, 'r') as f:
                    prediction = json.loads(f.read())
            else:
                prediction = np.load(custom_model_variant.outputPredictionsPath)

            # Load feature explorer predictions from file
            if hasattr(custom_model_variant, 'outputPredictionsFeatureExplorerPath'):
                feature_explorer_predictions = np.load(custom_model_variant.outputPredictionsFeatureExplorerPath)
                # Store each prediction with the original sample for the feature explorer
                if mode == 'classification':
                    prediction_samples = np.concatenate((Y_samples, np.array([feature_explorer_predictions.argmax(axis=1) + 1]).T), axis=1).tolist()
                elif mode == 'regression':
                    prediction_samples = np.concatenate((Y_samples, feature_explorer_predictions), axis=1).tolist()
        else:
            output_directory = os.path.dirname(os.path.realpath(model_path))
            # Make predictions
            if akida_model_path:
                prediction = ei_tensorflow.brainchip.model.make_predictions(
                    mode, akida_model_path, validation_dataset, Y_test,
                    num_classes, output_directory,
                    objdet_details=objdet_details)

                # for YOLOv2 the SoftMax was already applied in the output decoding function
                if objdet_details is None or objdet_details.last_layer != 'yolov2-akida':
                    # akida returns logits so apply softmax for y_pred_prob metrics
                    from scipy.special import softmax
                    prediction = softmax(prediction, axis=-1)

            else:
                prediction = make_predictions_tflite(mode, model, validation_dataset,
                                                    Y_test, num_classes, output_directory,
                                                    item_feature_axes, objdet_details)

    evaluator = Evaluator(per_sample_metadata, sample_id_details["validation"] if sample_id_details else None)

    if mode == 'classification':
        if class_names is None:
            raise ValueError('class_names must be provided for classification mode')
        eval_result = evaluator.classification(Y_test, prediction, class_names)
        if not is_custom_variant:
            prediction_samples = feature_explorer_predictions_classification(model, X_samples, Y_samples, item_feature_axes,
                                                                             has_samples, akida_model_path)

    elif mode == 'regression':
        # Unbatch predictions before passing in
        eval_result = evaluator.regression(Y_test, prediction[:, 0])

        if not is_custom_variant:
            prediction_samples = feature_explorer_predictions_regression(model, X_samples, Y_samples, item_feature_axes,
                                                                         has_samples)

    elif mode == 'object-detection':
        if objdet_details is None:
            raise ValueError('objdet_details must be provided for object-detection mode')

        # Derive width, height from ground truth
        if objdet_details.last_layer == "fomo":
            # For FOMO the data is not batched
            for image, _ in validation_dataset.take(1):
                width, height, _num_channels = image.shape
                break

            # TODO(mat): what should minimum_confidence_rating be here?
            y_pred_bbls = batch_convert_segmentation_map_to_object_detection_prediction(
                prediction, minimum_confidence_rating=0.5, fuse=True
            )

            # Do alignment by centroids. This results in a flatten list of int
            # labels that is suitable for confusion matrix calculations.
            # It will automatically add the implicit background class to the dataset.
            y_true_labels, y_pred_labels = dataset_match_by_near_centroids(
                # batch the data since the function expects it
                validation_dataset.batch(32, drop_remainder=False),
                y_pred_bbls,
                width,
            )
            eval_result = evaluator.fomo(
                class_names=class_names,
                y_true_labels=y_true_labels,
                y_pred_labels=y_pred_labels,
            )
        else:
            # note: for object detection always has extra dim ; (B, 1, W, H, C)
            for image, _ in validation_dataset:
                _batch, _single_instance, width, height, _num_channels = image.shape
                break
            y_true_bbls = BoundingBoxLabelScore.from_tf_dataset(validation_dataset)
            y_pred_bbls = BoundingBoxLabelScore.from_studio_predictions(prediction)

            eval_result = evaluator.object_detection(
                class_names=class_names,
                width=width,
                height=height,
                y_true_bbls=y_true_bbls,
                y_pred_bbls=y_pred_bbls,
            )

    elif mode == 'anomaly-gmm' or mode == 'visual-anomaly':
        # by definition we don't have any anomalies in the training dataset
        # so we don't calculate these metrics
        pass

    model_size = 0
    if model:
        model_size = len(model)

    if akida_model_path:
        is_supported_on_mcu = False
        mcu_support_error = "Akida models run only on Linux boards with AKD1000"
    elif not is_custom_variant:
        is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(model_file, log_messages=False)

    memory_async = None
    io_details = None

    if is_custom_variant and hasattr(custom_model_variant, 'outputProfilingPath'):
        # Load existing profiling metrics from file
        with open(custom_model_variant.outputProfilingPath, 'r') as f:
            model_info = json.loads(f.read())
            model_size = model_info['model_size']
            memory = model_info['memory']
            io_details = model_info['io_details']
            is_supported_on_mcu = False
            mcu_support_error = None
            memory_async = None
    elif (is_supported_on_mcu):
        if (not memory):
            # We will kick off a separate Docker container to calculate RAM/ROM.
            # Post-training the metadata is read (in studio/server/training/learn-block-keras.ts)
            # and any metrics that have `memoryAsync` will fire off a separate job (see handleAsyncMemory).
            # After the async memory is completed, the async memory job will overwrite the `memory` section
            # of the metadata.
            memory_async = {
                'type': 'requires-profiling',
            }
    else:
        memory = {}
        memory['tflite'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }
        memory['eon'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }

    model_info = {
        'type': model_type,
        'loss': eval_result.loss,
        'accuracy': eval_result.accuracy,
        'confusionMatrix': eval_result.matrix,
        'report': eval_result.report,
        'size': model_size,
        'estimatedMACCs': None,
        'memory': memory,
        'memoryAsync': memory_async,
        'predictions': prediction_samples,
        'isSupportedOnMcu': is_supported_on_mcu,
        'mcuSupportError': mcu_support_error,
        'metrics': eval_result.metrics,
    }

    if io_details is not None:
        model_info['io_details'] = io_details

    return model_info

def feature_explorer_predictions_classification(model, X_samples, Y_samples, item_feature_axes, has_samples, akida_model_path):
    """
    Generates predictions for the feature explorer in classification mode.
    """
    try:
        # Make predictions for feature explorer
        if has_samples:
            if model:
                feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples), item_feature_axes)
            elif akida_model_path:
                feature_explorer_predictions = ei_tensorflow.brainchip.model.predict(akida_model_path, X_samples, len(Y_samples))
            else:
                raise Exception('Expecting either a Keras model or an Akida model')

            # Store each prediction with the original sample for the feature explorer
            return np.concatenate((Y_samples, np.array([feature_explorer_predictions.argmax(axis=1) + 1]).T), axis=1).tolist()
    except Exception as e:
        print('Failed to generate feature explorer', e, flush=True)

def feature_explorer_predictions_regression(model, X_samples, Y_samples, item_feature_axes, has_samples):
    """
    Generates predictions for the feature explorer in regression mode.
    """
    try:
        # Make predictions for feature explorer
        if has_samples:
            feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples), item_feature_axes)
            # Store each prediction with the original sample for the feature explorer
            return np.concatenate((Y_samples, feature_explorer_predictions), axis=1).tolist()
    except Exception as e:
        print('Failed to generate feature explorer', e, flush=True)

def run_tasks_in_parallel(tasks, parallel_count):
    res = []
    with ThreadPoolExecutor(parallel_count) as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            res.append(running_task.result())
    return res

# Useful reference: https://machinethink.net/blog/how-fast-is-my-model/
def estimate_maccs_for_layer(layer):
    """Estimate the number of multiply-accumulates in a given Keras layer."""
    """Better than flops because there's hardware support for maccs."""
    if isinstance(layer, tf.keras.layers.Dense):
        # Ignore the batch dimension
        input_count = functools.reduce(operator.mul, layer.input.shape[1:], 1)
        return input_count * layer.units

    if (isinstance(layer, tf.keras.layers.Conv1D)
        or isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.Conv3D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        # The channel is either at the start or the end of the shape (ignoring)
        # the batch dimension
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
        else:
            input_channels = layer.input.shape[-1]
        # Ignore the batch dimension but include the channels
        output_size = functools.reduce(operator.mul, layer.output.shape[1:])
        return kernel_size * input_channels * output_size

    if (isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
            output_channels = layer.output.shape[1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[2:])
        else:
            input_channels = layer.input.shape[-1]
            output_channels = layer.output.shape[-1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[1:-1])
        # Calculate the MACCs for depthwise and pointwise steps
        depthwise_count = kernel_size * input_channels * output_size
        # If this is just a depthwise conv, we can return early
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return depthwise_count
        # Otherwise, calculate MACCs for the pointwise step and add them
        pointwise_count = input_channels * output_size * output_channels
        return depthwise_count + pointwise_count

    if isinstance(layer, tf.keras.Model):
        return estimate_maccs_for_model(layer)

    # For other layers just return 0. These are mostly stuff that doesn't involve MACCs
    # or stuff that isn't supported by TF Lite for Microcontrollers yet.
    return 0

def estimate_maccs_for_model(keras_model):
    maccs = 0

    # e.g. non-Keras saved model
    if not hasattr(keras_model, 'layers'):
        return maccs

    for layer in keras_model.layers:
        try:
            layer_maccs = estimate_maccs_for_layer(layer)
            maccs += layer_maccs
        except Exception as err:
            print('Error while estimating maccs for layer', flush=True)
            print(err, flush=True)
    return maccs

def describe_layers(keras_model):
    layers = []

    # e.g. non-Keras saved model
    if not hasattr(keras_model, 'layers'):
        return layers

    for l in range(len(keras_model.layers)):
        layer = keras_model.layers[l]
        input = layer.input
        if isinstance(input, list):
            input = input[0]
        layers.append({
            'input': {
                'shape': input.shape[1],
                'name': input.name,
                'type': str(input.dtype)
            },
            'output': {
                'shape': layer.output.shape[1],
                'name': layer.output.name,
                'type': str(layer.output.dtype)
            }
        })

    return layers


def get_recommended_model_type(float32_perf, int8_perf):
    # For now, always recommend int8 if available
    if int8_perf:
        return 'int8'
    else:
        return 'float32'

def get_model_metadata(keras_model: tf.keras.Model,
                       validation_dataset: tf.data.Dataset,
                       Y_test: np.ndarray,
                       X_samples: Optional[np.ndarray],
                       Y_samples: Optional[np.ndarray],
                       has_samples: bool,
                       class_names: List[str],
                       curr_model_metadata: Optional[Dict],
                       curr_model_metadata_mtime: Optional[int],
                       model_metadata_fname: str,
                       mode: ClassificationMode,
                       metrics_fname_prefix: str,
                       model_float32: Optional[bytes],
                       model_int8: Optional[bytes],
                       file_float32: str,
                       file_int8: Optional[str],
                       file_akida: Optional[str],
                       objdet_details: Optional[ObjectDetectionDetails],
                       per_sample_metadata: Optional[Dict],
                       # A dictionary containing the sample IDs in row order for each split (training and validation)
                       sample_id_details: Optional[Dict[str, List[int]]],
                       training_options_changed: bool,
                       custom_model_variants: Optional[List[CustomModelVariantInfo]] = None):

    metadata = {
        'metadataVersion': 5,
        'created': datetime.datetime.now().isoformat(),
        'classNames': class_names,
        'availableModelTypes': [],
        'recommendedModelType': '',
        'modelValidationMetrics': [],
        'modelIODetails': [],
        'mode': mode,
        'kerasJSON': None,
        'performance': None,
        'objectDetectionLastLayer': objdet_details.last_layer if objdet_details else None,
        'taoNMSAttributes': objdet_details.tao_nms_attributes if objdet_details else None,
    }

    # keep metadata from anomaly (gmm) training
    item_feature_axes = None
    if (
            curr_model_metadata and
            'mean' in curr_model_metadata and
            'scale' in curr_model_metadata and
            'axes' in curr_model_metadata and
            'defaultMinimumConfidenceRating' in curr_model_metadata
    ):
        metadata['mean'] = curr_model_metadata['mean']
        metadata['scale'] = curr_model_metadata['scale']
        metadata['axes'] = curr_model_metadata['axes']
        metadata['defaultMinimumConfidenceRating'] = curr_model_metadata['defaultMinimumConfidenceRating']
        item_feature_axes = metadata['axes']

    recalculate_memory = True
    recalculate_performance = True

    # e.g. ONNX conversion failed
    if (file_int8 and not os.path.exists(file_int8)):
        file_int8 = None

    # For some model types (e.g. object detection) there is no keras model, so
    # we are unable to compute some of our stats with these methods
    if keras_model:
        # This describes the basic inputs and outputs, but skips over complex parts
        # such as transfer learning base models
        metadata['layers'] = describe_layers(keras_model)
        estimated_maccs = estimate_maccs_for_model(keras_model)
        # This describes the full model, so use it to determine if the architecture
        # has changed between runs
        if hasattr(keras_model, 'to_json'):
            metadata['kerasJSON'] = keras_model.to_json()

        # Only recalculate memory when model architecture has changed
        if (
            curr_model_metadata and 'kerasJSON' in curr_model_metadata and 'metadataVersion' in curr_model_metadata
            and curr_model_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_model_metadata['kerasJSON']
        ):
            recalculate_memory = False
        else:
            recalculate_memory = True

        if (
            curr_model_metadata and 'kerasJSON' in curr_model_metadata and 'metadataVersion' in curr_model_metadata
            and curr_model_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_model_metadata['kerasJSON']
            and 'performance' in curr_model_metadata
            and curr_model_metadata['performance']
            # Re-compute performance when metadata is older than 2024-03-18 so EON RAM optimized performance
            # will be computed on re-train for existing models with no other changes to the model structure.
            and (curr_model_metadata_mtime is None or curr_model_metadata_mtime > 1710749753)
        ):
            metadata['performance'] = curr_model_metadata['performance']
            recalculate_performance = False
        else:
            recalculate_memory = True
            recalculate_performance = True

    else:
        metadata['layers'] = []
        estimated_maccs = -1
        # If there's no Keras model we can't tell if the architecture has changed, so recalculate memory every time
        recalculate_memory = True
        recalculate_performance = True

    # In non-multimodel cases, kerasJSON reflects the full model, so any changes indicate that the
    # model architecture has changed (see conditions above).
    # For visual AD, kerasJSON represents only the model backbone, so instead we check if the training
    # options have changed to also capture updates in scoring function parameters.
    if mode == 'visual-anomaly':
        RECALCULATION_THRESHOLD_TIMESTAMP = 1730505600  # Unix timestamp for 2024-11-02
        if training_options_changed:
            recalculate_memory = True
            recalculate_performance = True
        elif (curr_model_metadata_mtime is None or curr_model_metadata_mtime < RECALCULATION_THRESHOLD_TIMESTAMP):
            # Re-compute performance when metadata is older than the threshold timestamp so performance will
            # be computed on re-train for existing models with no other changes to the model structure,
            # as scoring head is now taken into account
            recalculate_memory = True
            recalculate_performance = True
        else:
            metadata['performance'] = curr_model_metadata['performance'] if curr_model_metadata else None
            recalculate_memory = False
            recalculate_performance = False

    # perf info remains valid? then keep deviceSpecificPerformance
    if not recalculate_memory and not recalculate_performance:
        if (curr_model_metadata and 'deviceSpecificPerformance' in curr_model_metadata):
            metadata['deviceSpecificPerformance'] = curr_model_metadata['deviceSpecificPerformance']

    if recalculate_performance:
        try:
            args = '/app/profiler/build/profiling '
            if file_float32:
                args = args + file_float32 + ' '
            if file_int8:
                args = args + file_int8 + ' '

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            if '{' in a and '}' in a:
                metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
                print('Calculating inferencing time OK', flush=True)
            else:
                print('Failed to calculate inferencing time:', a)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None

    float32_perf = None
    int8_perf = None

    # standalone full metrics made available for download from dashboard. always
    # reset metrics when running validation since model type may have changed
    # and we don't want to end up any stale float32/int8 results that might not
    # be rerun.
    metrics_json = MetricsJson(filename_prefix=metrics_fname_prefix, mode=mode, reset=True)

    if model_float32:
        try:
            print('Calculating float32 accuracy...', flush=True)
            model_type = 'float32'

            memory = None
            if not recalculate_memory and curr_model_metadata is not None:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_model_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            float32_perf = profile_model(
                model_type=model_type,
                model=model_float32,
                model_file=file_float32,
                akida_model_path=None,
                validation_dataset=validation_dataset,
                Y_test=Y_test,
                X_samples=X_samples,
                Y_samples=Y_samples,
                has_samples=has_samples,
                memory=memory,
                mode=mode,
                class_names=class_names,
                item_feature_axes=item_feature_axes,
                per_sample_metadata=per_sample_metadata,
                sample_id_details=sample_id_details,
                objdet_details=objdet_details)

            float32_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(float32_perf)
            metadata['modelIODetails'].append(get_io_details(model_float32, model_type))

            if 'metrics' in float32_perf:
                metrics_json.set('validation', model_type, float32_perf['metrics'])

        except Exception as err:
            print('Unable to execute TensorFlow Lite float32 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if model_int8:
        try:
            if file_int8 is None:
                raise Exception('Expecting a path to the int8 model file')
            print('Calculating int8 accuracy...', flush=True)
            model_type = 'int8'

            memory = None
            if not recalculate_memory and curr_model_metadata is not None:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_model_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            int8_perf = profile_model(
                model_type=model_type,
                model=model_int8,
                model_file=file_int8,
                akida_model_path=None,
                validation_dataset=validation_dataset,
                Y_test=Y_test,
                X_samples=X_samples,
                Y_samples=Y_samples,
                has_samples=has_samples,
                memory=memory,
                mode=mode,
                class_names=class_names,
                item_feature_axes=item_feature_axes,
                per_sample_metadata=per_sample_metadata,
                sample_id_details=sample_id_details,
                objdet_details=objdet_details)
            int8_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(int8_perf)
            metadata['modelIODetails'].append(get_io_details(model_int8, model_type))

            if 'metrics' in int8_perf:
                metrics_json.set('validation', model_type, int8_perf['metrics'])

        except Exception as err:
            print('Unable to execute TensorFlow Lite int8 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if file_akida:
        print('Profiling akida model...', flush=True)
        model_type = 'akida'

        program_size, total_nps, nodes = ei_tensorflow.brainchip.model.get_hardware_utilization(file_akida)
        flops, macs = ei_tensorflow.brainchip.model.get_macs_flops(keras_model)
        memory = {}
        memory['tflite'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        # only 'eon' is used, see comment in populateMetadataTemplate in
        # studio/client/project/pages/training-keras-ui.ts
        memory['eon'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        if model_int8:
            io_details = get_io_details(model_int8, model_type)
        else:
            io_details = None

        akida_perf = profile_model(
            model_type=model_type,
            model=None,
            model_file=None,
            akida_model_path=file_akida,
            validation_dataset=validation_dataset,
            Y_test=Y_test,
            X_samples=X_samples,
            Y_samples=Y_samples,
            has_samples=has_samples,
            memory=memory,
            mode=mode,
            class_names=class_names,
            item_feature_axes=None,
            per_sample_metadata=per_sample_metadata,
            sample_id_details=sample_id_details,
            objdet_details=objdet_details)
        sparsity = ei_tensorflow.brainchip.model.get_model_sparsity(file_akida, mode, validation_dataset,
                                                                    objdet_details=objdet_details)
        akida_perf['estimatedMACCs'] = macs
        metadata['availableModelTypes'].append(model_type)
        metadata['modelValidationMetrics'].append(akida_perf)

        if 'metrics' in akida_perf:
            metrics_json.set('validation', model_type, akida_perf['metrics'])

        # hack: let's grab input scailing from int8 model. Akida is expecting input to be only 8 bit
        # (or less - see Dense layer)
        if io_details is not None:
            metadata['modelIODetails'].append(io_details)

        # Also store against deviceSpecificPerformance
        metadata['deviceSpecificPerformance'] = {
            'brainchip-akd1000': {
                'model_quantized_int8_io.tflite': {
                    'latency': 0,
                    'ram': -1,
                    'rom': program_size,
                    'customMetrics': [
                        {
                            'name': 'nps',
                            'value': str(total_nps)
                        },
                        {
                            'name': 'sparsity',
                            'value': f'{sparsity:.2f}%'
                        },
                        {
                            'name': 'macs',
                            'value': str(int(macs))
                        }
                    ]
                }
            }
        }

    if custom_model_variants:
        for custom_variant in custom_model_variants:
            custom_variant_key = custom_variant.variant.key
            print('Profiling ' + custom_variant_key + ' model...', flush=True)
            variant_perf = profile_model(
                model_type=custom_variant_key,
                model=None,
                model_file=None,
                akida_model_path=file_akida,
                validation_dataset=validation_dataset,
                Y_test=Y_test,
                X_samples=X_samples,
                Y_samples=Y_samples,
                has_samples=has_samples,
                memory=memory,
                mode=mode,
                prepare_model_tflite_script=None,
                prepare_model_tflite_eon_script=None,
                class_names=class_names,
                item_feature_axes=None,
                async_memory_profiling=False,
                per_sample_metadata=per_sample_metadata,
                sample_id_details=sample_id_details,
                objdet_details=objdet_details,
                custom_model_variant=custom_variant)

            metadata['availableModelTypes'].append(custom_variant_key)
            metadata['modelValidationMetrics'].append(variant_perf)

            if 'metrics' in variant_perf:
                metrics_json.set('validation', custom_variant_key, variant_perf['metrics'])

            if 'io_details' in variant_perf:
                io_details = variant_perf['io_details']
                io_details['modelType'] = custom_variant_key
                metadata['modelIODetails'].append(io_details)

    # Decide which model to recommend
    if file_akida:
        metadata['recommendedModelType'] = 'akida'
    else:
        recommended_model_type = get_recommended_model_type(float32_perf, int8_perf)
        metadata['recommendedModelType'] = recommended_model_type

    with open(model_metadata_fname, 'w') as f:
        # Go through the metadata recursively and set any nan to None
        sanitized_data = sanitize_for_json(metadata)
        # Actually write the metadata to disk
        json.dump(sanitized_data, f, indent=4)

    return metadata
