#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .pairwise_field_features import *
from .type_detection import data_types, general_types
from .aggregation_helpers import *


def extract_aggregate_pairwise_field_features(pairwise_field_features):
    final_field_features = OrderedDict()

    for field_feature in all_pairwise_features_list:
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

    flattened_field_features = []

    for one_field_pairwise_field_features in pairwise_field_features:
        flattened_field_features.extend(one_field_pairwise_field_features)

    for field_feature in all_pairwise_features_list:
        feature_name = field_feature['name']
        feature_type = field_feature['type']

        if feature_type == 'id':
            continue

        field_feature_values = [
            f[feature_name] for f in flattened_field_features if f[feature_name] is not None]

        if not len(field_feature_values):
            continue

        if feature_type == 'boolean':
            for k, v in aggregate_boolean_features(field_feature_values, feature_name).items():
                final_field_features[k] = v

        if feature_type == 'numeric':
            for k, v in aggregate_numeric_features(field_feature_values, feature_name).items():
                final_field_features[k] = v

    return final_field_features
