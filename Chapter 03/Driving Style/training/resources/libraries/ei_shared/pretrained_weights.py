import os
from pathlib import Path
import requests
from typing import Optional, Tuple

def get_or_download_pretrained_weights(weights_prefix: str, num_channels: int, alpha: float, allowed_combinations: list) -> str:
    # Check if there's a dictionary in allowed_combinations with matching num_channels and alpha
    if not any(combination['num_channels'] == num_channels and combination['alpha'] == alpha for combination in allowed_combinations):
        raise Exception(
            f"Pretrained weights not currently available for num_channel={num_channels} with alpha={alpha}."
            f" Current supported combinations are {allowed_combinations}."
            " For further assistance please contact support at https://forum.edgeimpulse.com/"
        )

    weights_mapping = {
        (1, 0.1): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.grayscale.bsize_64.lr_0_05.epoch_441.val_loss_4.13.val_accuracy_0.2.hdf5",
        (1, 0.35): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5",
        (3, 0.1): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.color.bsize_64.lr_0_05.epoch_498.val_loss_3.85.hdf5",
        (3, 0.35): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5",
        (3, 1.0): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96.h5"
    }

    weights = os.path.join(weights_prefix, weights_mapping[(num_channels, alpha)])

    # Explicit check that requested weights are available.
    if (weights and not os.path.exists(weights)):
        p = Path(weights)
        if not p.exists():
            if not p.parent.exists():
                p.parent.mkdir(parents=True)
            root_url = 'https://cdn.edgeimpulse.com/'
            weights_data = requests.get(root_url + os.path.relpath(weights, weights_prefix)).content
            with open(weights, 'wb') as f:
                f.write(weights_data)

    return weights

def get_weights_path_if_available(weights_prefix: str, num_channels: int, alpha: float, dimension: int) -> Optional[Tuple[str, bool]]:
    # mappings with (weights_file_path, weights_include_top) tuples
    weights_mapping = {
        (1, 0.35, 96): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5", True),
        (1, 0.35, 128): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.128x128.grayscale.bsize_64.lr_0_05.epoch_99.val_loss_2.78_val_accuracy_0.41.h5", True),
        (1, 0.35, 160): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.160x160.grayscale.bsize_64.lr_0_05.epoch_75.val_loss_2.53_val_accuracy_0.45.h5", True),
        (1, 0.5, 96): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_5.96x96.grayscale.bsize_64.lr_0_05.epoch_61.val_loss_2.99_val_accuracy_0.37.h5", True),
        (1, 0.5, 128): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_5.128x128.grayscale.bsize_64.lr_0_05.epoch_63.val_loss_2.60_val_accuracy_0.44.h5", True),
        (1, 0.5, 160): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_5.160x160.grayscale.bsize_64.lr_0_05.epoch_79.val_loss_2.23_val_accuracy_0.50.h5", True),
        (1, 0.75, 96): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_75.96x96.grayscale.bsize_64.lr_0_05.epoch_72.val_loss_2.50_val_accuracy_0.45.h5", True),
        (1, 0.75, 128): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_75.128x128.grayscale.bsize_64.lr_0_05.epoch_70.val_loss_2.17_val_accuracy_0.51.h5", True),
        (1, 0.75, 160): ("transfer-learning-weights/edgeimpulse/MobileNetV2.0_75.160x160.grayscale.bsize_64.lr_0_05.epoch_79.val_loss_1.97_val_accuracy_0.55.h5", True),
        (1, 1.0, 96): ("transfer-learning-weights/edgeimpulse/MobileNetV2.1_0.96x96.grayscale.bsize_64.lr_0_05.epoch_66.val_loss_2.43_val_accuracy_0.47.h5", True),
        (1, 1.0, 128): ("transfer-learning-weights/edgeimpulse/MobileNetV2.1_0.128x128.grayscale.bsize_64.lr_0_05.epoch_44.val_loss_2.13_val_accuracy_0.52.h5", True),
        (1, 1.0, 160): ("transfer-learning-weights/edgeimpulse/MobileNetV2.1_0.160x160.grayscale.bsize_64.lr_0_05.epoch_78.val_loss_1.83_val_accuracy_0.58.h5", True),
        (3, 0.35, 224): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5", False),
        (3, 0.35, 192): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_192_no_top.h5", False),
        (3, 0.35, 160): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160_no_top.h5", False),
        (3, 0.35, 128): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5", False),
        (3, 0.35, 96): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96_no_top.h5", False),
        (3, 0.50, 128): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_128.h5", True),
        (3, 0.50, 160): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_160.h5", True),
        (3, 0.50, 192): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_192.h5", True),
        (3, 0.50, 224): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5", True),
        (3, 0.50, 96): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96.h5", True),
        (3, 0.75, 128): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_128.h5", True),
        (3, 0.75, 160): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_160.h5", True),
        (3, 0.75, 192): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_192.h5", True),
        (3, 0.75, 224): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5", True),
        (3, 0.75, 96): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_96.h5", True),
        (3, 1.00, 128): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128.h5", True),
        (3, 1.00, 160): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5", True),
        (3, 1.00, 192): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_192.h5", True),
        (3, 1.00, 224): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5", True),
        (3, 1.00, 96): ("transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96.h5", True),
    }

    try:
        weight_info = weights_mapping[(num_channels, alpha, dimension)]
        weights_path, weights_include_top = weight_info
        weights = os.path.join(weights_prefix, weights_path)
    except Exception as e:
        weights = None

    # return weights path if exists
    if (weights and os.path.exists(weights)):
        return weights, weights_include_top
    else:
        return None