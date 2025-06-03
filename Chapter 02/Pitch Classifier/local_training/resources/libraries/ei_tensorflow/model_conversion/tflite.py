import io, os
import tensorflow as tf
import numpy as np
import json
import traceback
from ..conversion import warn_about_issues, representative_dataset_generator, run_converter

def convert_int8_io_int8(tf_model_path, dataset_generator,
                                         tflite_file, disable_per_channel = False):
    converter_quantize = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    if disable_per_channel:
        converter_quantize._experimental_disable_per_channel = disable_per_channel
        print('    Note: Per channel quantization has been automatically disabled for this model.')
    converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quantize.representative_dataset = dataset_generator
    # Force the input and output to be int8
    converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
    converter_quantize.inference_input_type = tf.int8
    converter_quantize.inference_output_type = tf.int8
    tflite_quant_model = run_converter(converter_quantize)
    with open(tflite_file, 'wb') as f:
        f.write(tflite_quant_model)
    return tflite_quant_model

def convert_float32(tf_model_path, tflite_file):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter.target_spec.supported_types = [
        tf.dtypes.float32,
        tf.dtypes.int8
    ]
    tflite_model = run_converter(converter)
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    return tflite_model
