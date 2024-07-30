import pickle
import pandas as pd
import numpy as np
import random
import paddle
from collections import OrderedDict

def load_data(table_path, chart_path):
    with open(table_path, 'rb') as f:
        table_codes = pickle.load(f)
    with open(chart_path, 'rb') as f:
        chart_codes = pickle.load(f)
    return table_codes, chart_codes

def get_fids(data):
    fids = []
    for x in data:
        fids.append(x['fid'])
    return fids 

def is_all_none(src_list):
    return all(element is None for element in src_list)



def process_data(table_codes, chart_codes):
    final_data = []
    for i in range(len(table_codes)):

        table = table_codes[i]
        chart = chart_codes[i]
        assert table['fid']==chart['fid'] , 'table and chart not match'
        code = {}
        code['fid'] = table['fid']
        code['table'] = {}
        code['table']['head']=[]
        code['table']['value'] =[]
        code['format_table'] =''
        code['chart'] = ''
        try:
            for field_data in table['fields']:
                format_name = str(field_data['format_name'])
                code['table']['head'].append(format_name)
                code['table']['value'].append(field_data['values'][:200])
            
            
            v =code['table']['value']
            max_length = max(len(sublist) for sublist in v)
            if max_length==0:
                continue
            code['table']['value'] = [list(np.pad(sublist, (0, max_length - len(sublist)), mode='constant') )for sublist in v]
            
            
            
            s = '<begin><table_begin>'
            h = code['table']['head']
            v = code['table']['value']
            for i in range(len(v[0])):
                for j in range(len(v)):
                    s += str(h[j]) + ' is ' + str(v[j][i])+', '
                s = s[:-2]+' <table_end><end>'
            code['format_table'] = s
            
            c = {}
            c['encoding'] = {}
            c['encoding']['x'] = {}
            c['encoding']['y'] = []
            
            if is_all_none(chart['all_xsrc']) and not is_all_none(chart['all_ysrc']):
                for ind in range(len(chart['series_attr'])):
                    chart['series_attr'][ind]['format_xsrc'], chart['series_attr'][ind]['format_ysrc'] = chart['series_attr'][ind]['format_ysrc'],chart['series_attr'][ind]['format_xsrc']
        
            c['encoding']['x']['field'] = chart['series_attr'][0]['format_xsrc']
            
            for item in chart['series_attr']:
                j = {}
            
                j['field'] = item['format_ysrc']
                j['type'] = item['type']
                c['encoding']['y'].append(j)
            
            code['chart'] = c
            final_data.append(code)
        except Exception as e:
                print(e)
                continue
    return final_data


def split_data(final_data):
    line = []
    scatter = []
    bar = []
    his = []
    pie = []
    box = []
    for i in range(len(final_data)):
        item = final_data[i]
        if item['chart']['encoding']['y'][0]['type'] == 'scatter':
            scatter.append(item)
        if item['chart']['encoding']['y'][0]['type'] == 'pie':
            pie.append(item)
        if item['chart']['encoding']['y'][0]['type'] == 'line':
            line.append(item)
        if item['chart']['encoding']['y'][0]['type'] == 'box':
            box.append(item)
        if item['chart']['encoding']['y'][0]['type'] == 'bar':
            bar.append(item)
        if item['chart']['encoding']['y'][0]['type'] == 'histogram':
            his.append(item)
            
    random.shuffle(line)
    random.shuffle(scatter)
    random.shuffle(bar)
    random.shuffle(his)
    random.shuffle(pie)
    random.shuffle(box)

    def split_set(data):
        train_set = data[:int(len(data) * 0.8)]
        test_set = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        val_set = data[int(len(data) * 0.9):]
        return train_set, test_set, val_set

    train_set, test_set, val_set = [], [], []

    for dataset in [line, scatter, bar, his, pie, box]:
        t_train, t_test, t_val = split_set(dataset)
        train_set.extend(t_train)
        test_set.extend(t_test)
        val_set.extend(t_val)

    return train_set, test_set, val_set


def main():
    table_codes, chart_codes = load_data('processed_data/table_codes.pkl', 'processed_data/chart_codes.pkl')
    final_data = process_data(table_codes, chart_codes)
    train_set, test_set, val_set = split_data(final_data)

    train_fids, test_fids, val_fids = get_fids(train_set), get_fids(test_set), get_fids(val_set)


    paddle.save(final_data , 'input_data/final_data.pdparams')
    paddle.save(train_set , 'input_data/train_data.pdparams')
    paddle.save(test_set , 'input_data/test_data.pdparams')
    paddle.save(val_set , 'input_data/val_data.pdparams')
    paddle.save(train_fids , 'input_data/train_fids.pdparams')
    paddle.save(test_fids , 'input_data/test_fids.pdparams')
    paddle.save(val_fids , 'input_data/val_fids.pdparams')

   
if __name__ == "__main__":
    main()