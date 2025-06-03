import numpy as np
from typing import List, Union, Callable, Literal, Dict, Optional, Any
from collections import Counter
import os
import json
import math

from ei_shared.types import ClassificationMode


def parse_per_sample_metadata_ndjson(fname):
    """Helper to parse studio dump of sample to metadata.

    Args:
        fname: the per_sample_metadata ndjson file written by studio

    Returns:
        a dictionary mapping sample_id to metadata suitable as required
        by FacettedMetrics constructor
    """

    # TODO(mat) check what entry looks for like data point with no
    #  metadata; is it no entry? or empty entry?
    metadata = {}
    with open(fname, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            sample_id = data["id"]
            metadata[sample_id] = data["metadata"]
    return metadata


def allocate_to_bins(array: List, num_bins: int):
    """Given an array, and a number of bins, return a grouping across bins.

    There are two expected uses of this method ;

    1) determining a grouping for continuous values in regression metrics. e.g.

    groups = allocate_to_bins(y_pred, num_bins=3)
    calculate_regression_metrics(y_true, y_pred, groups=groups)

    2) grouping continuous meta data values into bins. e.g.

    groups = allocate_to_bins(continuous_meta_data_values, num_bins=3)
    calculate_regression_metrics(y_true, y_pred, groups=groups)

    Args:
        array: a list of numerical values
        num_bins: the number of bins to allocate across
    Returns:
        a grouping, the same length as input array, that can be used as the
        `groups` are for calculate_regression_metrics
    """

    array = np.array(array)

    # calculate the bin edges for a N bin histogram
    _bin_allocation, bin_edges = np.histogram(array, bins=num_bins)

    # allocate array values to groups based on bin_edges; i.e. the lowest
    # elements are in group 1, next elements are in group 2 etc.
    # since digitize uses >= this though results in the last bin having only
    # the max element. to avoid this we can ignore the last bin_edge; then
    # the max elements is dropped into the previous bin
    groups = np.digitize(array, bin_edges[:-1])

    # convert from numerical index into a human readable range
    # e.g. instead of, say, group=5 we have "(0.56, 0.67)"
    human_readable_groups = []
    for g in groups:
        range_min = bin_edges[g - 1]
        range_max = bin_edges[g]
        # Make sure we display enough decimal places to distinguish between
        # the min and max values, in case they are close in value.
        for decimal_places in range(2, 8):
            rounded_min = np.round(range_min, decimal_places)
            rounded_max = np.round(range_max, decimal_places)

            if not np.isclose(rounded_min, rounded_max):
                break
        human_readable_groups.append(f"({rounded_min}, {rounded_max})")

    return human_readable_groups


def calculate_grouped_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metrics_fn: Callable,
    groups: List,
    max_groups: Optional[int] = None,
    include_all: bool = True,
):
    """Given a metrics_fn and a grouping run the metrics_fn for each group.

    Args:
        y_true: complete set of y_true values, as either a list or ndarray
        y_pred: complete set of y_pred values, as either a list or ndarray
        metrics_fn: a callable that returns a dict of metrics for a (sub)set of
            y_true, y_pred values.
        groups: a list of items, the same length as y_true & y_pred that is
            used as a grouping key for metrics_fn calls
        max_groups: the maximum number of grouping that can be returned.
            included so code is robust to a (potentially) large number of
            distinct groups
        include_all: whether to include an entry for 'all' in the returned
            metrics

    Returns:
        A dictionary where metrics_fn has been called for the entire
        y_true, y_pred set as well as subsets of these based on the groups

    E.g. for arguments
        y_true = [3, 1, 4, 1, 5, 9]
        y_pred = [2, 6, 5, 3, 5, 8]
        metrics_fn = lambda(yt, yp): { 'max_t': max(yt), 'max_p': max(yp) }
        groups = ['a', 'a', 'a', 'b', 'c', 'c']

    the return would be
        {'all': { 'max_t': 9, 'max_p': 8 },
         'per_group': {
            'a': { 'max_t': 4, 'max_p': 6 },
            'b': { 'max_t': 1, 'max_p': 3 },
            'c': { 'max_t': 9, 'max_p': 8 },
         }
        }

    additionally if call was made with max_groups=2 then the entry for 'b'
    would not be included since the top 2 elements by frequency are 'a' & 'c'

    """

    # check sizes
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise Exception(
            "Expected lengths of y_true, y_pred and groups to be the"
            f" same but were {len(y_true)}, {len(y_pred)} and"
            f" {len(groups)} respectively"
        )

    # init returned metrics with an 'all' value
    metrics: dict[str, Any] = {"per_group": {}}
    if include_all:
        metrics["all"] = metrics_fn(y_true, y_pred)

    if max_groups is None:
        # no max_groups => use all groups
        filtered_groups = set(groups)
    else:
        # determine top groups by element frequency
        group_top_freqs = Counter(groups).most_common(max_groups)
        filtered_groups = [g for g, _freq in group_top_freqs]
        if set(groups) != set(filtered_groups):
            print(
                f"WARNING: filtering from {len(set(groups))} distinct groups"
                f" down to {len(set(filtered_groups))}"
            )

    # when y_true or y_pred are nd arrays we can efficiently slice out
    # a set of indexes with advanced indexing, otherwise we need to index
    # them out explicitly
    def extract_subset(a, idxs):
        if type(a) == np.ndarray:
            return a[idxs]
        elif type(a) == list:
            return [a[i] for i in idxs]
        else:
            raise TypeError(f"Expected ndarray or list, not {type(a)}")

    groups = np.array(groups)
    # Sort alphabetically so we have a stable order
    for group in sorted(list(filtered_groups)):
        idxs = np.where(groups == group)[0]
        y_true_subset = extract_subset(y_true, idxs)
        y_pred_subset = extract_subset(y_pred, idxs)
        metrics["per_group"][group] = metrics_fn(y_true_subset, y_pred_subset)

    return metrics


class MetricsJson(object):
    """Helper responsible for shared profiling and testing metrics json"""

    CURRENT_VERSION = 6

    def __init__(
        self,
        mode: ClassificationMode,
        filename_prefix: str,
        reset: bool = False,
    ):

        self.filename_prefix = filename_prefix
        self.mode = mode

        if (reset):
            dirname = os.path.dirname(filename_prefix)
            for filename in os.listdir(dirname):
                if (filename.startswith(os.path.basename(filename_prefix)) and filename.endswith('.json')):
                    os.remove(os.path.join(dirname, filename))

    def set(
        self,
        split: Literal["validation", "test"],
        model_type: Literal["float32", "int8", "akida"],
        metrics: Dict,
    ):
        data = {
            "version": MetricsJson.CURRENT_VERSION,
            "metrics": metrics
        }

        filename = self.filename_prefix + '_' + split + '_' + model_type + '.json'
        with open(filename, "w") as f:
            sanitized_data = sanitize_for_json(data)
            json.dump(sanitized_data, fp=f, indent=4)

def quantize_metadata(
    metadata: "dict[int, dict[str, str|int|float]]", num_bins: int = 4
):
    """
    Quantize metadata values to be used as input for FacettedMetrics. Numeric
    values are binned into human-readable groups, while non-numeric values are
    left as strings.

    Args:
        metadata: dictionary mapping sample_id to dictionary of { meta_data_key: meta_data_value, ... }.
        num_bins: the number of bins to allocate across for numeric values.
    """

    # Tracks whether a given metadata key is numeric or not
    metadata_key_numeric: Dict[str, bool] = {}

    # Collect all the values for each key
    metadata_values_by_key: Dict[str, List["str|int|float"]] = {}

    for sample_id, sample_metadata in metadata.items():
        # Look at every value to determine if this metadata is numeric.
        # If there's a single non-numeric value for a key, we'll consider
        # it non-numeric.
        for key, value in sample_metadata.items():
            # Store all the values for binning
            metadata_values_by_key.setdefault(key, []).append(value)
            if key not in metadata_key_numeric:
                metadata_key_numeric[key] = type(value) in [int, float]
            else:
                if metadata_key_numeric[key]:
                    if type(value) not in [int, float]:
                        metadata_key_numeric[key] = False

    # Quantize the numeric values
    for key, values in metadata_values_by_key.items():
        if metadata_key_numeric[key]:
            metadata_values_by_key[key] = allocate_to_bins(values, num_bins)

    # Produce a new version of the input metadata with the quantized values
    quantized_metadata: Dict[int, Dict[str, str]] = {}
    for sample_id, sample_metadata in metadata.items():
        quantized_metadata[sample_id] = {}
        for key, value in sample_metadata.items():
            if metadata_key_numeric[key]:
                # Substitute the actual values with the human-readable group strings
                quantized_metadata[sample_id][key] = str(
                    metadata_values_by_key[key].pop(0)
                )
            else:
                # If a key had any non-numeric values, ensure ALL of its values are strings
                quantized_metadata[sample_id][key] = str(value)

    return quantized_metadata

def sanitize_for_json(metadata: Dict):
    """
    Removes any NaN values from the metadata and converts them to None, since otherwise
    Python's JSON encoder will write nan values to the file, which is not valid JSON
    and causes problems when trying to read it from JSON compliant parsers.
    """
    def replace_nan_inf(d: Dict):
        # Recursive function to replace NaN, inf, and -inf with None
        if isinstance(d, dict):
            return {k: replace_nan_inf(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [replace_nan_inf(item) for item in d]
        elif isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
            return None
        else:
            return d
    return replace_nan_inf(metadata)
