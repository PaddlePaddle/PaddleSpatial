#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment
import pandas as pd
from collections import OrderedDict

from .helpers import *
from .type_detection import detect_field_type, data_type_to_general_type, data_types, general_types

# np.warnings.filterwarnings('ignore')

field_basic_features_list = [
    {'name': 'fid', 'type': 'id'},
    {'name': 'field_id', 'type': 'id'},
    {'name': 'exists', 'type': 'boolean'},
    {'name': 'length', 'type': 'numeric'}
]

for data_type in data_types:
    field_basic_features_list.append({
        'name': 'data_type_is_{}'.format(data_type), 'type': 'boolean'
    })

for general_type in general_types:
    field_basic_features_list.append({
        'name': 'general_type_is_{}'.format(general_type), 'type': 'boolean'
    })

field_existence_features_list = [
    {'name': 'has_none', 'type': 'boolean'},
    {'name': 'percentage_none', 'type': 'numeric'},
    {'name': 'num_none', 'type': 'numeric'},
]

field_uniqueness_features_list = [
    {'name': 'num_unique_elements', 'type': 'numeric'},
    {'name': 'unique_percent', 'type': 'numeric'},
    {'name': 'is_unique', 'type': 'boolean'}
]

field_c_statistical_features_list = [
    {'name': 'list_entropy', 'type': 'numeric'},
    {'name': 'mean_value_length', 'type': 'numeric'},
    {'name': 'median_value_length', 'type': 'numeric'},
    {'name': 'min_value_length', 'type': 'numeric'},
    {'name': 'max_value_length', 'type': 'numeric'},
    {'name': 'std_value_length', 'type': 'numeric'},
    {'name': 'percentage_of_mode', 'type': 'numeric'},
]

field_q_statistical_features_list = [
    {'name': 'mean', 'type': 'numeric'},
    {'name': 'normalized_mean', 'type': 'numeric'},
    {'name': 'median', 'type': 'numeric'},
    {'name': 'normalized_median', 'type': 'numeric'},

    {'name': 'var', 'type': 'numeric'},
    {'name': 'std', 'type': 'numeric'},
    {'name': 'coeff_var', 'type': 'numeric'},
    {'name': 'min', 'type': 'numeric'},
    {'name': 'max', 'type': 'numeric'},
    {'name': 'range', 'type': 'numeric'},
    {'name': 'normalized_range', 'type': 'numeric'},

    {'name': 'entropy', 'type': 'numeric'},
    {'name': 'gini', 'type': 'numeric'},
    {'name': 'q25', 'type': 'numeric'},
    {'name': 'q75', 'type': 'numeric'},
    {'name': 'med_abs_dev', 'type': 'numeric'},
    {'name': 'avg_abs_dev', 'type': 'numeric'},
    {'name': 'quant_coeff_disp', 'type': 'numeric'},
    {'name': 'skewness', 'type': 'numeric'},
    {'name': 'kurtosis', 'type': 'numeric'},
    {'name': 'moment_5', 'type': 'numeric'},
    {'name': 'moment_6', 'type': 'numeric'},
    {'name': 'moment_7', 'type': 'numeric'},
    {'name': 'moment_8', 'type': 'numeric'},
    {'name': 'moment_9', 'type': 'numeric'},
    {'name': 'moment_10', 'type': 'numeric'},

    {'name': 'percent_outliers_15iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_3iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_1_99', 'type': 'numeric'},
    {'name': 'percent_outliers_3std', 'type': 'numeric'},
    {'name': 'has_outliers_15iqr', 'type': 'boolean'},
    {'name': 'has_outliers_3iqr', 'type': 'boolean'},
    {'name': 'has_outliers_1_99', 'type': 'boolean'},
    {'name': 'has_outliers_3std', 'type': 'boolean'},
    {'name': 'normality_statistic', 'type': 'numeric'},
    {'name': 'normality_p', 'type': 'numeric'},
    {'name': 'is_normal_5', 'type': 'boolean'},
    {'name': 'is_normal_1', 'type': 'boolean'},
]


field_name_features_list = [
    {'name': 'field_name_length', 'type': 'numeric'},
    {'name': 'x_in_name', 'type': 'boolean'},
    {'name': 'y_in_name', 'type': 'boolean'},
    {'name': 'id_in_name', 'type': 'boolean'},
    {'name': 'time_in_name', 'type': 'boolean'},
    {'name': 'digit_in_name', 'type': 'boolean'},
    {'name': 'dollar_in_name', 'type': 'boolean'},
    {'name': 'pound_in_name', 'type': 'boolean'},
    {'name': 'euro_in_name', 'type': 'boolean'},
    {'name': 'yen_in_name', 'type': 'boolean'},
    {'name': 'first_char_uppercase_name', 'type': 'boolean'},
    {'name': 'num_uppercase_characters', 'type': 'numeric'},
    {'name': 'space_in_name', 'type': 'boolean'},
    {'name': 'number_of_words_in_name', 'type': 'numeric'},
]

field_sequence_features_list = [
    {'name': 'is_sorted', 'type': 'boolean'},
    {'name': 'is_monotonic', 'type': 'boolean'},
    {'name': 'sortedness', 'type': 'numeric'},

    {'name': 'lin_space_sequence_coeff', 'type': 'numeric'},
    {'name': 'log_space_sequence_coeff', 'type': 'numeric'},
    {'name': 'is_lin_space', 'type': 'boolean'},
    {'name': 'is_log_space', 'type': 'boolean'},
]

all_field_features_list = \
    field_basic_features_list + \
    field_existence_features_list + \
    field_uniqueness_features_list + \
    field_c_statistical_features_list + \
    field_q_statistical_features_list + \
    field_name_features_list + \
    field_sequence_features_list

all_field_features_list_names = [x['name'] for x in all_field_features_list]


def get_existence_features(v):
    r = OrderedDict([(f['name'], None) for f in field_existence_features_list])
    if not len(v):
        return r
    num_none = sum(1 for e in v if (pd.isnull(e)))
    r['num_none'] = num_none
    r['percentage_none'] = num_none / len(v)
    r['has_none'] = (num_none > 0)

    return r

# Sequence Properties


def get_uniqueness_features(v, field_type, field_general_type):
    r = OrderedDict([(f['name'], None)
                     for f in field_uniqueness_features_list])
    if not len(v):
        return r
    if field_general_type == 'c' or field_type == 'integer':
        unique_elements = get_unique(v)
        r = {}
        r['num_unique_elements'] = unique_elements.size
        r['unique_percent'] = (r['num_unique_elements'] / len(v))
        r['is_unique'] = (r['num_unique_elements'] == len(v))
    return r


def get_statistical_features(v, field_type, field_general_type):
    r = OrderedDict([(f['name'], None)
                     for f in field_c_statistical_features_list + field_q_statistical_features_list])

    if not len(v):
        return r
    if field_general_type == 'c':
        r['list_entropy'] = list_entropy(v)

        value_lengths = [len(x) for x in v]
        r['mean_value_length'] = np.mean(value_lengths)
        r['median_value_length'] = np.median(value_lengths)
        r['min_value_length'] = np.min(value_lengths)
        r['max_value_length'] = np.max(value_lengths)
        r['std_value_length'] = np.std(value_lengths)
        r['percentage_of_mode'] = (pd.Series(v).value_counts().max() / len(v))

    if field_general_type in 'q':
        sample_mean = np.mean(v)
        sample_median = np.median(v)
        sample_var = np.var(v)
        sample_min = np.min(v)
        sample_max = np.max(v)
        sample_std = np.std(v)
        q1, q25, q75, q99 = np.percentile(v, [0.01, 0.25, 0.75, 0.99])
        iqr = q75 - q25

        r['mean'] = sample_mean
        r['normalized_mean'] = sample_mean / sample_max
        r['median'] = sample_median
        r['normalized_median'] = sample_median / sample_max

        r['var'] = sample_var
        r['std'] = sample_std
        r['coeff_var'] = (sample_mean / sample_var) if sample_var else None
        r['min'] = sample_min
        r['max'] = sample_max
        r['range'] = r['max'] - r['min']
        r['normalized_range'] = (r['max'] - r['min']) / \
            sample_mean if sample_mean else None

        r['entropy'] = entropy(v)
        r['gini'] = gini(v)
        r['q25'] = q25
        r['q75'] = q75
        r['med_abs_dev'] = np.median(np.absolute(v - sample_median))
        r['avg_abs_dev'] = np.mean(np.absolute(v - sample_mean))
        r['quant_coeff_disp'] = (q75 - q25) / (q75 + q25)
        r['coeff_var'] = sample_var / sample_mean
        r['skewness'] = skew(v)
        r['kurtosis'] = kurtosis(v)
        r['moment_5'] = moment(v, moment=5)
        r['moment_6'] = moment(v, moment=6)
        r['moment_7'] = moment(v, moment=7)
        r['moment_8'] = moment(v, moment=8)
        r['moment_9'] = moment(v, moment=9)
        r['moment_10'] = moment(v, moment=10)

        # Outliers
        outliers_15iqr = np.logical_or(
            v < (q25 - 1.5 * iqr), v > (q75 + 1.5 * iqr))
        outliers_3iqr = np.logical_or(v < (q25 - 3 * iqr), v > (q75 + 3 * iqr))
        outliers_1_99 = np.logical_or(v < q1, v > q99)
        outliers_3std = np.logical_or(
            v < (
                sample_mean -
                3 *
                sample_std),
            v > (
                sample_mean +
                3 *
                sample_std))
        r['percent_outliers_15iqr'] = np.sum(outliers_15iqr) / len(v)
        r['percent_outliers_3iqr'] = np.sum(outliers_3iqr) / len(v)
        r['percent_outliers_1_99'] = np.sum(outliers_1_99) / len(v)
        r['percent_outliers_3std'] = np.sum(outliers_3std) / len(v)

        r['has_outliers_15iqr'] = np.any(outliers_15iqr)
        r['has_outliers_3iqr'] = np.any(outliers_3iqr)
        r['has_outliers_1_99'] = np.any(outliers_1_99)
        r['has_outliers_3std'] = np.any(outliers_3std)

        # Statistical Distribution
        if len(v) >= 8:
            normality_k2, normality_p = normaltest(v)
            r['normality_statistic'] = normality_k2
            r['normality_p'] = normality_p
            r['is_normal_5'] = (normality_p < 0.05)
            r['is_normal_1'] = (normality_p < 0.01)

    return r


def get_name_features(n):
    r = OrderedDict([(f['name'], None) for f in field_name_features_list])
    r['field_name_length'] = len(n)
    r['x_in_name'] = (n.startswith('x') or n.endswith('x'))
    r['y_in_name'] = (n.startswith('y') or n.endswith('y'))
    r['id_in_name'] = (n.startswith('id') or n.endswith('id'))
    r['time_in_name'] = ('time' in n)
    r['digit_in_name'] = bool(re.match(r'^(?=.*\d).+$', n))
    r['dollar_in_name'] = ('$' in n)
    r['pound_in_name'] = ('£' in n)
    r['euro_in_name'] = ('€' in n)
    r['yen_in_name'] = ('¥' in n)
    r['first_char_uppercase_name'] = (n[0] == n[0].upper())

    num_uppercase_characters = sum(1 for c in n if c.isupper())

    r['num_uppercase_characters'] = num_uppercase_characters
    r['space_in_name'] = (' ' in n)
    r['number_of_words_in_name'] = len(n.split(' '))

    return r


def get_sequence_features(v, field_type, field_general_type):
    r = OrderedDict([(f['name'], None) for f in field_sequence_features_list])
    if not len(v):
        return r
    sorted_v = np.sort(v)

    if field_general_type == 'c':
        r['is_sorted'] = np.array_equal(sorted_v, v)

    if field_general_type == 't':
        v = v.astype('int')
        sorted_v = sorted_v.astype('int')
    if field_general_type in ['t', 'q']:
        sequence_incremental_subtraction = np.subtract(
            sorted_v[:-1], sorted_v[1:]).astype(int)
        r['is_monotonic'] = np.all(
            sequence_incremental_subtraction <= 0) or np.all(
            sequence_incremental_subtraction >= 0)
        r['sortedness'] = np.absolute(
            pearsonr(v, sorted_v)[0])  # or use inversions
        # np.allclose(v, sorted_v)  # np.array_equal(sorted_v, v)
        r['is_sorted'] = np.array_equal(sorted_v, v)
    if field_general_type == 'q':
        sequence_incremental_division = np.divide(sorted_v[:-1], sorted_v[1:])
        sequence_incremental_subtraction = np.diff(sorted_v)
        r['lin_space_sequence_coeff'] = np.std(
            sequence_incremental_subtraction) / np.mean(sequence_incremental_subtraction)
        r['log_space_sequence_coeff'] = np.std(
            sequence_incremental_division) / np.mean(sequence_incremental_division)
        r['is_lin_space'] = r['lin_space_sequence_coeff'] <= 0.001
        r['is_log_space'] = r['log_space_sequence_coeff'] <= 0.001
    return r


def extract_single_field_features(fields, fid, timeout=15, MAX_FIELDS=10):
    all_field_features = []
    for i in range(0, MAX_FIELDS):
        single_field_feature_set = OrderedDict()
        for f in all_field_features_list:
            if f['type'] == 'boolean':
                single_field_feature_set[f['name']] = False
            else:
                single_field_feature_set[f['name']] = None
        all_field_features.append(single_field_feature_set)

    parsed_fields = []  # For pairwise feature extraction
    for i, (field_name, d) in enumerate(fields):
        field_id = d['uid']
        field_order = d['order']
        field_values = d['data']

        field_length = len(field_values)
        field_type, field_scores = detect_field_type(field_values)
        field_general_type = data_type_to_general_type[field_type]

        all_field_features[i]['fid'] = fid
        all_field_features[i]['field_id'] = '{}:{}'.format(
            fid.split(':')[0] if fid else None, field_id)
        all_field_features[i]['exists'] = True
        all_field_features[i]['length'] = field_length
        # all_field_features[i]['data_type'] = field_type
        # all_field_features[i]['general_type'] = field_general_type
        all_field_features[i]['data_type_is_{}'.format(field_type)] = True
        all_field_features[i]['general_type_is_{}'.format(
            field_general_type)] = True

        parsed_field = {
            'name': field_name,
            'order': field_order,
            'data_type': field_type,
            'general_type': field_general_type
        }
        existence_features = OrderedDict()
        uniqueness_features = OrderedDict()
        statistical_features = OrderedDict()
        name_features = OrderedDict()
        sequence_features = OrderedDict()
        try:
            v = parse(field_values, field_type, field_general_type)
            v = np.ma.array(v).compressed()

            parsed_field['data'] = v
            parsed_field['unique_data'] = get_unique(v)

            start_time = time()
            while time() < (start_time + timeout):
                existence_features = get_existence_features(field_values)
                uniqueness_features = get_uniqueness_features(
                    v, field_type, field_general_type)
                statistical_features = get_statistical_features(
                    v, field_type, field_general_type)
                name_features = get_name_features(field_name)
                sequence_features = get_sequence_features(
                    v, field_type, field_general_type)
                break
        except Exception as e:
            print(e)
            print('Error parsing {}: {}'.format(field_name, e))
            pass

        for feature_set in [uniqueness_features, existence_features,
                            statistical_features, name_features, sequence_features]:
            for k, v in feature_set.items():
                all_field_features[i][k] = v
        parsed_fields.append(parsed_field)

    return all_field_features, parsed_fields





