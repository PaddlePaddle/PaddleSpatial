import json
import pandas as pd
import os
import numpy as np


def clean_chunk(chunk):
    # Filtering
    df_final_rows = []
    for i, x in chunk.iterrows():
        try:
            chart_data = json.loads(x.chart_data)
            layout = json.loads(x.layout)
            table_data = json.loads(x.table_data)

            if not (bool(chart_data) and bool(table_data)):
                continue

            df_final_rows.append({
            'fid': x['fid'],
            'chart_data': chart_data,
            'layout': layout,
            'table_data': table_data
            })

        except Exception as e:
            print(e)
            continue
            
    return pd.DataFrame(df_final_rows)