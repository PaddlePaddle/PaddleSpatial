#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Include boolean fields (cf Tableau?)
'''
import operator
import numpy as np
from time import time, strftime
from collections import OrderedDict, Counter

from itertools import groupby, combinations
from .type_detection import *
from scipy.stats import entropy
import scipy.cluster.hierarchy as hcluster
import editdistance


def calculate_overlap(a_data, b_data):
    a_min, a_max = np.min(a_data), np.max(a_data)
    a_range = a_max - a_min
    b_min, b_max = np.min(b_data), np.max(b_data)
    b_range = b_max - b_min
    has_overlap = False
    overlap_percent = 0
    if (a_max >= b_min) and (b_min >= a_min):
        has_overlap = True
        overlap = (a_max - b_min)
    if (b_max >= a_min) and (a_min >= b_min):
        has_overlap = True
        overlap = (b_max - a_min)
    if has_overlap:
        overlap_percent = max(overlap / a_range, overlap / b_range)
    if ((b_max >= a_max) and (b_min <= a_min)) or (
            (a_max >= b_max) and (a_min <= b_min)):
        has_overlap = True
        overlap_percent = 1
    return has_overlap, overlap_percent


def madm(a):
    # should be faster to not use masked arrays.
    a = np.ma.array(a).compressed()
    med = np.median(a)
    return np.median(np.abs(a - med))


def get_shared_elements(v1, v2):
    return [e for e in v1 if v1 in v2]


def get_unique(li, preserve_order=False):
    if preserve_order:
        seen = set()
        seen_add = seen.add
        return [x for x in li if not (x in seen or seen_add(x))]
    else:
        return np.unique(li)


def get_list_uniqueness(l):
    if len(l):
        return len(np.unique(l)) / len(l)
    else:
        return None


def detect_unique_list(l, THRESHOLD=0.95):
    if len(l) and ((len(np.unique(l)) / len(l)) >= THRESHOLD):
        return True
    return False


def list_entropy(l):
    return entropy(pd.Series(l).value_counts() / len(l))


def parse(v, field_type, field_general_type, drop=True):
    # Get rid of None
    if drop:
        v = np.array([e for e in v if e is not None])
    else:
        v = np.array(v)

    if field_type == 'integer':
        # localized_v = [ locale.atoi(e) for e in v]
        try:
            return v.astype(np.integer)
        except ValueError as ve:
            result = []
            for e in v:
                try:
                    result.append(int(e))
                except TypeError as e:
                    raise e
                except ValueError as e:
                    continue
            return result

    if field_type == 'decimal':
        # localized_v = [ locale.atof(e) for e in v ]
        try:
            return v.astype(np.integer)
        except ValueError as ve:
            result = []
            for e in v:
                try:
                    result.append(float(e))
                except TypeError as e:
                    raise e
                except ValueError as e:
                    continue
            return result

    if field_type == 'time':
        try:
            return pd.to_datetime(
                v,
                errors='coerce',
                infer_datetime_format=True,
                utc=True
            )
        except Exception as e:
            print('Cannot cast to time', v, e)
            return v

    return v


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array = np.add(array, 0.0000001)  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) /
            (n * np.sum(array)))  # Gini coefficient


def get_q_vector_features(n, v):
    q25, q75 = np.percentile(v, [0.25, 0.75])
    var = np.var(v)
    mean = np.mean(v)
    v_min = np.min(v)
    v_max = np.max(v)
    v_range = v_max - v_min
    r = OrderedDict()
    r['{}_min'.format(n)] = v_min
    r['{}_median'.format(n)] = np.median(v)
    r['{}_mean'.format(n)] = mean
    r['{}_max'.format(n)] = v_max
    r['{}_var'.format(n)] = var
    r['{}_cv'.format(n)] = var / mean
    r['{}_madm'.format(n)] = madm(v)
    r['{}_qcd'.format(n)] = (q75 - q25) / (q75 + q25)
    r['{}_normalized_field_length_range'.format(n)] = v_range / v_max
    return r
