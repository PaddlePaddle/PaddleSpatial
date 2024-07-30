import numpy as np
from .single_field_features import all_field_features_list
from .pairwise_field_features import all_pairwise_features_list


feature_name_to_type = {}
for x in all_field_features_list + all_pairwise_features_list:
    feature_name_to_type[x['name']] = x['type']

# Add square, log, squareroot to all numeric features
supplemental_functions = {
    'log': np.log,
    'square': np.square,
    'sqrt': np.sqrt
}


def supplement_features(d):
    supplemented_d = {}

    for k, v in d.items():
        supplemented_d[k] = v

        if k in ['fid', 'field_id']:
            continue

        feature_name = k.split('-agg-')[0]
        feature_type = feature_name_to_type.get(feature_name)

        if feature_name in ['data_type_entropy', 'general_type_entropy']:
            feature_type = 'numeric'

        if feature_type == 'numeric':
            for supplemental_function_name in supplemental_functions.keys():
                supplemented_d['{}-trans-{}'.format(k,
                                                    supplemental_function_name)] = None

            for supplemental_function_name, supplemental_function in supplemental_functions.items():
                try:
                    supplemented_d['{}-trans-{}'.format(
                        k, supplemental_function_name)] = supplemental_function(v)
                except Exception as e:
                    continue
    return supplemented_d
