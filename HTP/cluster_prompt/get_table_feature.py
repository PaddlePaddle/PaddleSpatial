import pandas as pd
from feature_extraction.features.aggregate_pairwise_field_features import extract_aggregate_pairwise_field_features
from feature_extraction.features.aggregate_single_field_features import extract_aggregate_single_field_features
from feature_extraction.features.pairwise_field_features import extract_pairwise_field_features
from feature_extraction.features.single_field_features import extract_single_field_features
from feature_extraction.outcomes.chart_outcomes import extract_chart_outcomes
from feature_extraction.outcomes.field_encoding_outcomes import extract_field_outcomes
from feature_extraction.features.transform import supplement_features
from feature_extraction.general_helpers import load_raw_data, clean_chunk, persist_features
import traceback
import pandas as pd

import os
import json
from time import time, strftime

import multiprocessing
from multiprocessing import Pool
from collections import OrderedDict, Counter
from pprint import pprint

compute_features_config = {
    'single_field': True,
    'aggregate_single_field': True,

    'pairwise_field': True,
    'aggregate_pairwise_field': True,

    'field_level_features': False,
    'chart_outcomes': False,
    'field_outcomes': False,
    'supplement': False,
}

def extract_features_from_fields(
        fields, compute_features_config, chart_obj={}, fid=None):
    results = {}
    if len(fields) <10:
        MAX_FIELDS = len(fields)
    else:
        MAX_FIELDS = 10
        fields = fields[:10]
    feature_names_by_type = {
        'basic': ['fid'],
        'single_field': [],
        'aggregate_single_field': [],
        'pairwise_field': [],
        'aggregate_pairwise_field': [],
        'chart_outcomes': [],
        'field_outcomes': []
    }

    df_feature_tuples_if_exists = OrderedDict({'fid': fid})
    df_feature_tuples = OrderedDict({'fid': fid})
    df_outcomes_tuples = OrderedDict()

    if compute_features_config['single_field'] or compute_features_config['field_level_features']:
        single_field_features, parsed_fields = extract_single_field_features(
            fields, fid, MAX_FIELDS=MAX_FIELDS)

        for i, field_features in enumerate(single_field_features):
            field_num = i + 1

            for field_feature_name, field_feature_value in field_features.items():
                if field_feature_name not in ['fid', 'field_id']:
                    field_feature_name_with_num = '{}_{}'.format(
                        field_feature_name, field_num)
                    if field_features['exists']:
                        df_feature_tuples_if_exists[field_feature_name_with_num] = field_feature_value

        if compute_features_config['field_level_features']:
            df_field_level_features = []
            for i, f in enumerate(single_field_features):
                if f['exists']:
                    if compute_features_config['supplement']:
                        f = supplement_features(f)
                    df_field_level_features.append(f)
                    feature_names_by_type['single_field'] = list(f.keys())
            results['df_field_level_features'] = df_field_level_features
        results['single_field_features'] = single_field_features

    if compute_features_config['aggregate_single_field']:
        aggregate_single_field_features = extract_aggregate_single_field_features(
            single_field_features
        )

        if compute_features_config['supplement']:
            aggregate_single_field_features = supplement_features(
                aggregate_single_field_features)

        for k, v in aggregate_single_field_features.items():
            df_feature_tuples[k] = v
            df_feature_tuples_if_exists[k] = v
            feature_names_by_type['aggregate_single_field'].append(k)

        results['aggregate_single_field_features'] = aggregate_single_field_features

    if compute_features_config['pairwise_field'] or compute_features_config['aggregate_pairwise_field']:
        pairwise_field_features = extract_pairwise_field_features(
            parsed_fields,
            single_field_features,
            fid,
            MAX_FIELDS=MAX_FIELDS
        )

        results['pairwise_field_features'] = pairwise_field_features

    if compute_features_config['aggregate_pairwise_field']:
        aggregate_pairwise_field_features = extract_aggregate_pairwise_field_features(
            pairwise_field_features)

        if compute_features_config['supplement']:
            aggregate_pairwise_field_features = supplement_features(
                aggregate_pairwise_field_features)

        for k, v in aggregate_pairwise_field_features.items():
            df_feature_tuples[k] = v
            df_feature_tuples_if_exists[k] = v
            feature_names_by_type['aggregate_pairwise_field'].append(k)

        results['aggregate_pairwise_field_features'] = aggregate_pairwise_field_features

    if compute_features_config['chart_outcomes']:
        outcomes = extract_chart_outcomes(chart_obj)
        for k, v in outcomes.items():
            df_outcomes_tuples[k] = v
            feature_names_by_type['chart_outcomes'].append(k)

    if compute_features_config['field_outcomes']:
        field_level_outcomes = extract_field_outcomes(chart_obj)
        feature_names_by_type['field_outcomes'] = list(
            list(field_level_outcomes)[0].keys())
        results['df_field_level_outcomes'] = field_level_outcomes

    results['df_feature_tuples'] = df_feature_tuples
    results['df_feature_tuples_if_exists'] = df_feature_tuples_if_exists
    results['df_outcomes_tuples'] = df_outcomes_tuples
    results['feature_names_by_type'] = feature_names_by_type

    return results

data = pd.read_csv('../data/origin_data.tsv',sep='\t')

clean_tdf = clean_chunk(data)


CHUNK_SIZE = 1000
dataset_feature_list = []

for chart_num, chart_obj in clean_tdf.iterrows():
    print(chart_num)
    fid = chart_obj.fid
    table_data = chart_obj.table_data

    fields = table_data[list(table_data.keys())[0]]['cols']
    sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
    num_fields = len(sorted_fields)
    extraction_results = extract_features_from_fields(
                sorted_fields, compute_features_config, chart_obj=chart_obj, fid=fid)
    dataset_feature_list.append(extraction_results['df_feature_tuples'])

df = pd.DataFrame(dataset_feature_list)
df.to_csv('features.tsv',sep='\t', index=False)