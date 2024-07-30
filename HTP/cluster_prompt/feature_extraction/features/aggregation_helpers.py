import numpy as np
import scipy as sc
from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment
import pandas as pd
from collections import OrderedDict

c_aggregation_functions = [
    'num',
    'has',
    'only_one',
    'all',
    'percentage',
]

q_aggregation_functions = [
    'mean',
    'var',
    'std',
    'avg_abs_dev',
    'med_abs_dev',
    'coeff_var',
    'min',
    'max',
    'range',
    'normalized_range',
]


def aggregate_boolean_features(v, feature_name):
    r = OrderedDict([('{}-agg-{}'.format(feature_name, c_aggregation_function), None)
                     for c_aggregation_function in c_aggregation_functions])

    r['{}-agg-num'.format(feature_name)] = sum(v)
    try:
        r['{}-agg-has'.format(feature_name)] = any(v)
        r['{}-agg-only_one'.format(feature_name)] = (sum(v) == 1)
        r['{}-agg-all'.format(feature_name)] = all(v)
    except Exception as e:
        print(feature_name, e, len(v), v)

    r['{}-agg-percentage'.format(feature_name)] = sum(v) / \
        len(v) if len(v) else None

    return r


def aggregate_numeric_features(v, feature_name):
    r = OrderedDict([('{}-agg-{}'.format(feature_name, q_aggregation_function), None)
                     for q_aggregation_function in q_aggregation_functions])
    v = np.asarray(v)
    sample_min = np.min(v)
    sample_max = np.max(v)
    sample_mean = np.mean(v)
    sample_median = np.median(v)
    sample_std = np.std(v)
    sample_var = np.var(v)
    r['{}-agg-mean'.format(feature_name)] = sample_mean
    r['{}-agg-var'.format(feature_name)] = sample_var
    r['{}-agg-std'.format(feature_name)] = sample_std
    r['{}-agg-avg_abs_dev'.format(feature_name)
      ] = np.mean(np.absolute(v - sample_mean))
    r['{}-agg-med_abs_dev'.format(feature_name)
      ] = np.median(np.absolute(v - sample_median))
    r['{}-agg-coeff_var'.format(feature_name)] = (sample_mean /
                                                  sample_var) if sample_var else None
    r['{}-agg-min'.format(feature_name)] = sample_min
    r['{}-agg-max'.format(feature_name)] = sample_max
    r['{}-agg-range'.format(feature_name)] = sample_max - \
        sample_min if (sample_max and sample_min) else None
    r['{}-agg-normalized_range'.format(feature_name)] = (sample_max - sample_min) / \
        sample_mean if (sample_max and sample_min and sample_mean) else None

    return r
