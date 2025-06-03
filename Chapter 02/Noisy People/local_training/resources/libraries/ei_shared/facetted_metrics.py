from collections import Counter, defaultdict
import scipy
from typing import Callable, List, Optional, Union
import numpy as np

from ei_shared.metrics_utils import calculate_grouped_metrics


class FacettedMetrics(object):

    def _log(self, *msgs):
        if self.logger is not None:
            # don't construct string version until we know logging is enabled
            msg = "".join(map(str, msgs))
            self.logger.info(msg)

    def __init__(
        self,
        per_sample_metadata: dict,
        loss_fn: Callable,
        stats_test_name: str,
        max_meta_data_values: int = 20,
        logger=None,
    ):
        """
        Args
            per_sample_metadata: dictionary mapping sample_id to dictionary
                of { meta_data_key: meta_data_value, ... }. it's expected the
                meta data values have been quantised as required ( e.g. by
                a call to allocate_to_bins )
            loss_fn: function mapping y_true and y_pred to a loss value
            stats_test_name: statistical test to use for comparing pairs of
                sets of loss values; i.e. 'ttest_ind' or 'kruskal'
            max_meta_data_values : maximum number of distinct values that
                are checked for each meta data key during grouping. These
                values chosen by highest frequency of value. values outside
                this set are rolled up into value "_OTHER", instances
                without metadata are rolled up into value "_UNSET"
        """

        self.per_sample_metadata = per_sample_metadata
        self.loss_fn = loss_fn
        self.stats_test_name = stats_test_name
        self.max_meta_data_values = max_meta_data_values
        self.logger = logger
        self._calculate_meta_data_distinct_values_by_key()

    def _calculate_meta_data_distinct_values_by_key(self):
        # count the frequency of meta data values, per key, from across the entire
        # known dataset
        self.meta_data_distinct_values_by_key = defaultdict(Counter)
        for sample_meta_data in self.per_sample_metadata.values():
            for key, value in sample_meta_data.items():
                self.meta_data_distinct_values_by_key[key].update([value])

    def _derive_keys_to_process(self):
        # check how counts of distinct meta data values are distributed.
        # the two extremes can be ignored for facetting...
        # 1) if all values are unique ( e.g. date time stamp, sequence number etc) no need to do grouping
        # 2) if all values are the same no need to do grouping either
        # we could additionally also ignore very low frequency items since
        # their stats are likely unstable but since we return a support we can
        # leave them in and just process top N by frequency.
        keys_to_process = set()
        for key, distinct_values in self.meta_data_distinct_values_by_key.items():
            if len(distinct_values) == len(self.per_sample_metadata):
                self._log(
                    "IGNORE [", key, "] has all distinct values => no need to group"
                )
            elif len(distinct_values) == 1:
                self._log(
                    "IGNORE [",
                    key,
                    "] only has single value for all data; => no need to group",
                )
            else:
                keys_to_process.add(key)
        # Sort so that the order is stable
        return sorted(list(keys_to_process))

    def _grouping_for_key(self, key, sample_ids):
        # build a grouping suitable for a call to calculate_grouped_metrics
        # recall: grouping is a list with a value per instance. meta data values
        # outside the top N by frequency, or values where meta data is unset, are
        # rolled up to _OTHER and _UNSET respectively

        # we first map from sample_id to meta_data value for key
        grouping = []
        for sample_id in sample_ids:
            if sample_id not in self.per_sample_metadata:
                grouping.append("_UNSET")
                continue
            sample_metadata = self.per_sample_metadata[sample_id]
            if key in sample_metadata:
                grouping.append(sample_metadata[key])
            else:
                grouping.append("_UNSET")

        # secondly, if configured, we take only the top N by frequency and
        # roll all others into a _OTHER value. this is to avoid having to
        # do a large number of unneccesary stats tests for the lower frequency
        # items
        if self.max_meta_data_values is not None:
            grouping_freqs = Counter(grouping)
            top_N_values_frequencies = grouping_freqs.most_common(
                self.max_meta_data_values
            )
            top_N_values = set([value for value, _freq in top_N_values_frequencies])
            grouping = [g if g in top_N_values else "_OTHER" for g in grouping]

        return grouping

    def _stats_test_fn(self, a, b):
        # TODO: do this in constructor
        if self.stats_test_name == "ttest_ind":
            return scipy.stats.ttest_ind(
                a, b, permutations=100, equal_var=True, random_state=42
            )
        elif self.stats_test_name == "kruskal":
            return scipy.stats.kruskal(a, b)
        else:
            raise Exception(
                f"Expected --stat_test to be in ['ttest_ind',"
                f"'kruskal'] but was {self.stats_test_name}"
            )

    def run_test(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        row_to_sample_id: List[int],
        meta_data_keys_to_process: Optional["set[str]"] = None,
    ):
        """
        Args:
            y_true: y true values
            y_pred: predicted values
            row_to_sample_id: list of sample_ids, the same size as y_true & y_pred
            meta_data_keys_to_process: if set only run test for this key,
                otherwise derive which keys to check based on statistics of the
                meta data value
        """

        # check args
        if y_true is np.ndarray and y_pred is np.ndarray:
            if y_true.shape != y_pred.shape:
                raise Exception("shapes of y_true and y_pred don't match")
        elif len(y_true) != len(y_pred):
            raise Exception("lengths of y_true and y_pred don't match")
        if len(row_to_sample_id) != len(y_true):
            raise Exception(
                f"|row_to_sample_id| = {len(row_to_sample_id)}"
                f" but needs to be |y_true| = |y_pred| = {len(y_true)}"
            )

        # if keys to process was passed as an arg we can use it, otherwise
        # derive a set of keys to process based on metadata stats.
        if meta_data_keys_to_process is None:
            meta_data_keys_to_process = self._derive_keys_to_process()

        self._log("meta_data_keys_to_process", meta_data_keys_to_process)

        # keep a list of all results, will sort at end by the stat
        results = []
        for meta_data_key in meta_data_keys_to_process:

            self._log("running tests for ", meta_data_key)

            # calculate grouped xent values
            grouping = self._grouping_for_key(meta_data_key, row_to_sample_id)
            self._log("grouping ", grouping)
            cgm = calculate_grouped_metrics(
                y_true=y_true, y_pred=y_pred, metrics_fn=self.loss_fn, groups=grouping
            )

            # run test for each subgroup
            for subgroup_key in cgm["per_group"].keys():

                self._log("running test for subgroup_key ", subgroup_key)

                if subgroup_key == "_OTHER":
                    # no need to collect stats on _OTHER, these were
                    # a subset only introduced to keep a cap on distinct values
                    continue

                subgroup = cgm["per_group"][subgroup_key]
                self._log("subgroup ", subgroup)

                # avoid small samples; these are not only unstable but
                # invalid for cases of len=1
                if len(subgroup) <= 3:
                    self._log(
                        "ignoring [",
                        meta_data_key,
                        "]/[",
                        subgroup_key,
                        "]; only ",
                        len(subgroup),
                        " values",
                    )
                    continue

                # the test is always this subgroup vs everything else
                all_without_subgroup = list(cgm["all"].copy())
                for e in subgroup:
                    all_without_subgroup.remove(e)
                self._log("all_without_subgroup ", all_without_subgroup)

                # collect result
                result = self._stats_test_fn(subgroup, all_without_subgroup)
                self._log("result ", result)

                results.append(
                    {
                        "key": meta_data_key,
                        "subgroup": subgroup_key,
                        "statistic": result.statistic,
                        "pvalue": result.pvalue,
                        "support": len(subgroup),
                    }
                )

        # Sort by statistic, then key and subgroup in case of ties
        return sorted(results, key=lambda e: (-abs(e["statistic"]), e["key"], e["subgroup"]))
