#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .single_field_features import *
from .type_detection import data_types, general_types
from .aggregation_helpers import *


def extract_aggregate_single_field_features(single_field_features):
    final_field_features = OrderedDict()

    for field_feature in all_field_features_list:
        feature_name = field_feature['name']
        feature_type = field_feature['type']
        if feature_type == 'boolean':
            for c_aggregation_function in c_aggregation_functions:
                final_field_features['{}-agg-{}'.format(
                    feature_name, c_aggregation_function)] = None
        if feature_type == 'numeric':
            for q_aggregation_function in q_aggregation_functions:
                final_field_features['{}-agg-{}'.format(
                    feature_name, q_aggregation_function)] = None

    for field_feature in all_field_features_list:
        feature_name = field_feature['name']
        feature_type = field_feature['type']

        if feature_type == 'id':
            continue

        field_feature_values = [
            f[feature_name] for f in single_field_features if f[feature_name] is not None]
        if not len(field_feature_values):
            continue

        if feature_type == 'boolean':
            for k, v in aggregate_boolean_features(
                    field_feature_values, feature_name).items():
                final_field_features[k] = v

        if feature_type == 'numeric':
            for k, v in aggregate_numeric_features(
                    field_feature_values, feature_name).items():
                final_field_features[k] = v

    dataset_field_types = []
    dataset_general_types = []
    # Special aggregations
    for f in single_field_features:
        if not f['exists']:
            continue

        for field_type in data_types:
            if f['data_type_is_{}'.format(field_type)]:
                dataset_field_types.append(field_type)

        for general_type in general_types:
            if f['general_type_is_{}'.format(general_type)]:
                dataset_general_types.append(general_type)

    final_field_features['data_type_entropy'] = list_entropy(
        dataset_field_types)
    final_field_features['general_type_entropy'] = list_entropy(
        dataset_general_types)

    return final_field_features
