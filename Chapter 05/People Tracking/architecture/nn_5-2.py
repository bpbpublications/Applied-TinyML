sys.path.append('./resources/libraries')
import os
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Conv2D, Softmax, Reshape
from tensorflow.keras.models import Model
from ei_tensorflow.constrained_object_detection import models, dataset, metrics, util
from ei_tensorflow.velo import train_keras_model_with_velo
from ei_shared.pretrained_weights import get_or_download_pretrained_weights
import ei_tensorflow.training

WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())

def build_model(input_shape: tuple, weights: str, alpha: float,
                num_classes: int) -> tf.keras.Model:
    """ Construct a constrained object detection model.

    Args:
        input_shape: Passed to MobileNet construction.
        weights: Weights for initialization of MobileNet where None implies
            random initialization.
        alpha: MobileNet alpha value.
        num_classes: Number of classes, i.e. final dimension size, in output.

    Returns:
        Uncompiled keras model.

    Model takes (B, H, W, C) input and
    returns (B, H//8, W//8, num_classes) logits.
    """

    #! First create full mobile_net_V2 from (HW, HW, C) input
    #! to (HW/8, HW/8, C) output
    mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                weights=weights,
                                alpha=alpha,
                                include_top=True)
    #! Default batch norm is configured for huge networks, let's speed it up
    for layer in mobile_net_v2.layers:
        if type(layer) == BatchNormalization:
            layer.momentum = 0.9
    #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
    cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
    #! Now attach a small additional head on the MobileNet
    model = Conv2D(filters=32, kernel_size=1, strides=1,
                activation='relu', name='head')(cut_point.output)
    logits = Conv2D(filters=num_classes, kernel_size=1, strides=1,
                    activation=None, name='logits')(model)
    return Model(inputs=mobile_net_v2.input, outputs=logits)

def train(num_classes: int, learning_rate: float, num_epochs: int,
          alpha: float, object_weight: float,
          train_dataset: tf.data.Dataset,
          validation_dataset: tf.data.Dataset,
          best_model_path: str,
          input_shape: tuple,
          batch_size: int,
          use_velo: bool = False,
          ensure_determinism: bool = False) -> tf.keras.Model:
    """ Construct and train a constrained object detection model.

    Args:
        num_classes: Number of classes in datasets. This does not include
            implied background class introduced by segmentation map dataset
            conversion.
        learning_rate: Learning rate for Adam.
        num_epochs: Number of epochs passed to model.fit
        alpha: Alpha used to construct MobileNet. Pretrained weights will be
            used if there is a matching set.
        object_weight: The weighting to give the object in the loss function
            where background has an implied weight of 1.0.
        train_dataset: Training dataset of (x, (bbox, one_hot_y))
        validation_dataset: Validation dataset of (x, (bbox, one_hot_y))
        best_model_path: location to save best model path. note: weights
            will be restored from this path based on best val_f1 score.
        input_shape: The shape of the model's input
        batch_size: Training batch size
        ensure_determinism: If true, functions that may be non-
            deterministic are disabled (e.g. autotuning prefetch). This
            should be true in test environments.
    Returns:
        Trained keras model.

    Constructs a new constrained object detection model with num_classes+1
    outputs (denoting the classes with an implied background class of 0).
    Both training and validation datasets are adapted from
    (x, (bbox, one_hot_y)) to (x, segmentation_map). Model is trained with a
    custom weighted cross entropy function.
    """

    nonlocal callbacks

    num_classes_with_background = num_classes + 1

    width, height, input_num_channels = input_shape
    if width != height:
        raise Exception(f"Only square inputs are supported; not {input_shape}")

    #! Use pretrained weights, if we have them for configured
    allowed_combinations = [{'num_channels': 1, 'alpha': 0.1},
                            {'num_channels': 1, 'alpha': 0.35},
                            {'num_channels': 3, 'alpha': 0.1},
                            {'num_channels': 3, 'alpha': 0.35}]
    weights = get_or_download_pretrained_weights(WEIGHTS_PREFIX, input_num_channels, alpha, allowed_combinations)

    model = build_model(
        input_shape=input_shape,
        weights=weights,
        alpha=alpha,
        num_classes=num_classes_with_background
    )

    #! Derive output size from model
    model_output_shape = model.layers[-1].output.shape
    _batch, width, height, num_classes = model_output_shape
    if width != height:
        raise Exception(f"Only square outputs are supported; not {model_output_shape}")
    output_width_height = width

    #! Build weighted cross entropy loss specific to this model size
    weighted_xent = models.construct_weighted_xent_fn(model.output.shape, object_weight)

    prefetch_policy = 1 if ensure_determinism else tf.data.experimental.AUTOTUNE

    #! Transform bounding box labels into segmentation maps
    def as_segmentation(ds, shuffle):
        ds = ds.map(dataset.bbox_to_segmentation(output_width_height, num_classes_with_background))
        if not ensure_determinism and shuffle:
            ds = ds.shuffle(buffer_size=batch_size*4)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(prefetch_policy)
        return ds

    train_segmentation_dataset = as_segmentation(train_dataset, True)
    validation_segmentation_dataset = as_segmentation(validation_dataset, False)

    validation_dataset_for_callback = (validation_dataset
        .batch(batch_size, drop_remainder=False)
        .prefetch(prefetch_policy))

    #! Initialise bias of final classifier based on training data prior.
    util.set_classifier_biases_from_dataset(
        model, train_segmentation_dataset)

    if not use_velo:
        model.compile(loss=weighted_xent,
                    optimizer=Adam(learning_rate=learning_rate))

    #! Create callback that will do centroid scoring on end of epoch against
    #! validation data. Include a callback to show % progress in slow cases.
    callbacks = callbacks if callbacks else []
    callbacks.append(metrics.CentroidScoring(validation_dataset_for_callback,
                                             output_width_height, num_classes_with_background))
    callbacks.append(metrics.PrintPercentageTrained(num_epochs))

    #! Include a callback for model checkpointing based on the best validation f1.
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(best_model_path,
            monitor='val_f1', save_best_only=True, mode='max',
            save_weights_only=True, verbose=0))

    if use_velo:
        from tensorflow.python.framework.errors_impl import ResourceExhaustedError
        try:
            train_keras_model_with_velo(
                model,
                train_segmentation_dataset,
                validation_segmentation_dataset,
                loss_fn=weighted_xent,
                num_epochs=num_epochs,
                callbacks=callbacks
            )
        except ResourceExhaustedError as e:
            print(str(e))
            raise Exception(
                "ResourceExhaustedError caught during train_keras_model_with_velo."
                " Though VeLO encourages a large batch size, the current"
                f" size of {batch_size} may be too large. Please try a lower"
                " value. For further assistance please contact support"
                " at https://forum.edgeimpulse.com/")
    else:
        model.fit(train_segmentation_dataset,
                validation_data=validation_segmentation_dataset,
                epochs=num_epochs, callbacks=callbacks, verbose=0)

    #! Restore best weights.
    model.load_weights(best_model_path)

    #! Add explicit softmax layer before export.
    softmax_layer = Softmax()(model.layers[-1].output)
    model = Model(model.input, softmax_layer)

    return model


EPOCHS = args.epochs or 60
LEARNING_RATE = args.learning_rate or 0.001
BATCH_SIZE = args.batch_size or 32

model = train(num_classes=classes,
              learning_rate=LEARNING_RATE,
              num_epochs=EPOCHS,
              alpha=0.35,
              object_weight=100,
              train_dataset=train_dataset,
              validation_dataset=validation_dataset,
              best_model_path=BEST_MODEL_PATH,
              input_shape=MODEL_INPUT_SHAPE,
              batch_size=BATCH_SIZE,
              use_velo=False,
              ensure_determinism=ensure_determinism)



disable_per_channel_quantization = False
