import numpy as np
import pandas as pd
import json
import os
import json


def load_raw_data(fp, chunk_size=1000):
    print('Loading raw data from %s' % fp)
    df = pd.read_table(
        fp,
        chunksize=chunk_size,
        encoding='utf-8'
    )
    return df



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

    '''if field_type == 'time':
        try:
            return pd.to_datetime(
                v,
                errors='coerce',
                infer_datetime_format=True,
                utc=True
            )
        except Exception as e:
            print('Cannot cast to time', v, e)
            return v'''

    return v