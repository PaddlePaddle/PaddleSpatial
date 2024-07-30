#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
from time import time, strftime
from pprint import pprint
import pickle


def load_raw_data(chunk_size=1000):
    data_file_name = '/data2/lixinhang/baidu/p40/data2charts/origin_train.tsv'
    print('Loading raw data from %s' % data_file_name)

    df = pd.read_table(
        data_file_name,
        chunksize=chunk_size,
        encoding='utf-8'
    )
    return df


def clean_chunk(chunk):
    # Filtering
    df_final_rows = []
    errors = 0
    empty_fields = 0
    global charts_without_data
    global chart_loading_errors

    for i, x in chunk.iterrows():
        try:
            chart_data = json.loads(x.chart_data)
            layout = json.loads(x.layout)
            table_data = json.loads(x.table_data)

            # Filter empty fields
            if not (bool(chart_data) and bool(table_data)):
                empty_fields += 1

                charts_without_data += 1
                chart_loading_errors += 1
                continue

            df_final_rows.append({
                'fid': x['fid'],
                'chart_data': chart_data,
                'layout': layout,
                'table_data': table_data
            })

        except Exception as e:
            errors += 1
            print(e)
            continue

    return pd.DataFrame(df_final_rows)


def persist_features(extraction_results, max_fields, features_dir_name,
                     feature_names_by_type={}, write_header=False):
    print('Persisting features')
    print('With header', write_header)

    print(feature_names_by_type)
    print(extraction_results['features_df'].shape)
    print('Number of features')
    for i, x in feature_names_by_type.items():
        print(i, len(x))

    for i, features_df in enumerate(
            extraction_results['features_df_by_num_fields']):
        num_fields = i + 1
        output_file_name = os.path.join(
            features_dir_name,
            'by_field',
            'features_{}.csv'.format(num_fields))
        features_df.to_csv(
            output_file_name,
            mode='a',
            index=False,
            header=write_header)

    for i, outcomes_df in enumerate(
            extraction_results['outcomes_df_by_num_fields']):
        num_fields = i + 1
        output_file_name = os.path.join(
            features_dir_name,
            'by_field',
            'outcomes_{}.csv'.format(num_fields))
        outcomes_df.to_csv(
            output_file_name,
            mode='a',
            index=False,
            header=write_header)

    if feature_names_by_type:
        pickle.dump(
            feature_names_by_type,
            open(
                os.path.join(
                    features_dir_name,
                    'feature_names_by_type.pkl'),
                'wb'))

    datasets = [
        {'key': 'features_df', 'filename': 'features_aggregate_single.csv',
            'subset': feature_names_by_type['basic'] + feature_names_by_type['aggregate_single_field']},
        {'key': 'features_df', 'filename': 'features_aggregate_single_pairwise.csv',
            'subset': feature_names_by_type['basic'] + feature_names_by_type['aggregate_single_field'] + feature_names_by_type['aggregate_pairwise_field']},
        {'key': 'outcomes_df', 'filename': 'chart_outcomes.csv'},
        {'key': 'field_level_features_df', 'filename': 'field_level_features.csv'},
        {'key': 'field_level_outcomes_df', 'filename': 'field_level_outcomes.csv'}
    ]

    for dataset in datasets:
        df = extraction_results.get(dataset['key'])
        if dataset.get('subset'):
            df = df[dataset['subset']]
        output_file_name = os.path.join(features_dir_name, dataset['filename'])
        df.to_csv(output_file_name, mode='a', index=False, header=write_header)
