import pandas as pd
import numpy as np
import json
import pickle
from utils.load import load_raw_data, parse
from utils.clean_data import clean_chunk
from utils.detect_type import detect_field_type, data_type_to_general_type, check_if_line_chart 
from utils.chart_attr import *
from utils.layout_attr import *



TYPES = ['scatter','line','bar','pie','histogram','box']




CHUNK_SIZE  = 1000



def one_axis_side(xsrcs,ysrcs):
    if (len(set(xsrcs))==1) | (len(set(ysrcs))==1):
        return True
    else:
        return False

def get_one_trace_type(trace_types):
    if len(set(trace_types))==1:
        return trace_types[0]
    else:
        return None

def get_chart_attr(chart_data):
    all_xsrc, all_ysrc, all_zsrc, trace_type =[], [], [], []
    series_attr_list = []
    for (num,d) in enumerate(chart_data):
    
        series_attr = {}

        series_type = get_type(d)
        series_name = get_series_name(d)
        xsrc = get_xsrc(d, series_type)
        ysrc = get_ysrc(d, series_type)
        zsrc = get_zsrc(d, series_type)

        width, color, bins_num, size = get_add_attr(series_type, d)

        series_attr['id'] = num
        series_attr['type'] = series_type
        series_attr['name'] = series_name
        series_attr['xsrc'] = xsrc
        series_attr['ysrc'] = ysrc
        series_attr['zsrc'] = zsrc
        series_attr['width'] = width
        series_attr['color'] = color
        series_attr['bins_num'] = bins_num
        series_attr['size'] = size

        all_xsrc.append(xsrc)
        all_ysrc.append(ysrc)
        all_zsrc.append(zsrc)
        trace_type.append(series_type)
        series_attr_list.append(series_attr)
    return all_xsrc, all_ysrc, all_zsrc, trace_type, series_attr_list


def get_layout_attr(layout):
    
    title = get_title(layout)
    barmode = get_barmode(layout)
    width = get_width(layout)
    height = get_height(layout)
    xrange = get_xrange(layout)
    yrange = get_yrange(layout)
    xtitle = get_xtitle(layout)
    ytitle = get_ytitle(layout)
    ztitle = get_ztitle(layout)
    
    return title, barmode, width, height, xrange, yrange, xtitle, ytitle,ztitle



def get_attr(chart_data, layout):
    flag=  1
    all_xsrc, all_ysrc, all_zsrc, trace_types, series_attr_list = get_chart_attr(chart_data)
    one_trace_type = get_one_trace_type(trace_types)
    is_one_axis_side = one_axis_side(all_xsrc,all_ysrc)

    if one_trace_type not in TYPES:
        flag = 0

    if is_one_axis_side:
        xsrc = all_xsrc[0]
    else:
        flag = 0
        xsrc = None
    layout_attr = get_layout_attr(layout)
    
    return flag, all_xsrc, all_ysrc, all_zsrc, trace_types, series_attr_list, one_trace_type, is_one_axis_side,xsrc, layout_attr

def generate_field_types(table):
    data_labels = {"str": 0, "num": 0, "dt": 0}
    field_name_types = {}
    field_name_types_array = {}
    fid = table['fid']
    for field in table['fields']:
        if field['type'] == 'q':
            replace_num_var = "num" + str(data_labels["num"])
            data_labels["num"] = data_labels["num"] + 1
            field_name_types_array[fid.split(':')[0] + ':' + field['uid']] = replace_num_var
        elif field['type'] == 't':
            replace_num_var = "dt" + str(data_labels["dt"])
            data_labels["dt"] = data_labels["dt"] + 1
            field_name_types_array[fid.split(':')[0] + ':' + field['uid']] = replace_num_var
        else:
            replace_num_var = "str" + str(data_labels["str"])
            data_labels["str"] = data_labels["str"] + 1
            field_name_types_array[fid.split(':')[0] + ':' + field['uid']] = replace_num_var
    return field_name_types_array

def replace_fieldnames(table, field_name_types_array, replace_direction):
    for field_name in field_name_types_array:
        field = list(field_name.keys())[0]
        value = field_name[field]
        
        if replace_direction == "z":
            table = str(table).replace(str(field),value)
        else:
            table = str(table).replace(str(value),field)
    return table

def get_chart_code(fid, chart_data, layout):
    flag, all_xsrc, all_ysrc,  all_zsrc, trace_types, series_attr_list, one_trace_type, is_one_x_side,xsrc, layout_attr = get_attr(chart_data,layout)
    title, barmode, width, height, xrange, yrange, xtitle, ytitle,ztitle = layout_attr
    
    if not flag:
        return None

    y_num = len(series_attr_list)
    pp = {}
    encoding_series  = []

    pp['fid'] = fid
    pp['all_xsrc'] = all_xsrc
    pp['all_ysrc'] = all_ysrc
    pp['all_zsrc'] = all_zsrc
    pp['trace_types'] = trace_types
    pp['one_trace_type'] = one_trace_type
    pp['is_one_x_side'] = is_one_x_side
    pp['xtitle'], pp['ytitle'], pp['ztitle'], pp['title'] = xtitle, ytitle, ztitle, title


    for i in range(y_num):
        attr = series_attr_list[i]
        t = {}
        t['xsrc'] = attr['xsrc']
        t['ysrc'] = attr['ysrc']
        t['zsrc'] = attr['zsrc']
        t['type'] = attr['type']
        t['name'] = attr['name']
        t['width'] = attr['width']
        t['color'] = attr['color']
        t['bins_num'] = attr['bins_num']
        t['size'] = attr['size']
        t['barmode'] = barmode

        encoding_series.append(t)
    
    pp['series_attr'] = encoding_series
    
    return pp

def get_format_table(fid, sorted_fields):
    table = {}
    table['fid'] = fid
    table['fields'] = []
    table_value = []
    for i, (field_name, d) in enumerate(sorted_fields):
        field = {}
        field_id = d['uid']
        field_order = d['order']
        field_values = d['data']

        field_type, field_scores = detect_field_type(field_values)
        field_general_type = data_type_to_general_type[field_type]

        try:
            v = parse(field_values, field_type, field_general_type)
            v = np.ma.array(v).compressed()
            if field_general_type == 'c':
                v = list(v.astype('str'))
            elif field_general_type == 't':
                v = [ np.datetime_as_string(x)  for x in v] 
            else:
                v = list(v.astype('float'))
        except Exception as e:
            print('Error parsing {}: {}'.format(field_name, e))
            continue
        field['uid'] = field_id
        field['name'] = field_name
        field['values'] = list(v)
        table_value.append(list(v))
        field['type'] = field_general_type
        table['fields'].append(field)
    return table, table_value


def get_table_code(fid, table_data):
    fields = table_data[list(table_data.keys())[0]]['cols']
    sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
    table, table_value = get_format_table(fid, sorted_fields)
    return table,table_value

def convert(table,new_table,chart):
    field_name_types_array = generate_field_types(table)
    table_code = replace_fieldnames(new_table, field_name_types_array, 'z')
    chart_code = replace_fieldnames(chart, field_name_types_array, 'z')
    return table_code, chart_code


def add_format(table,chart,field_name_types_dict):
    fid = table['fid']
    for i in range(len(table['fields'])):
        table['fields'][i]['owner:uid'] = fid.split(':')[0] + ':' + table['fields'][i]['uid']
        table['fields'][i]['format_name'] = field_name_types_dict[fid.split(':')[0] + ':' + table['fields'][i]['uid']]
    
    for i in range(len(chart['series_attr'])):
        if chart['series_attr'][i]['xsrc'] == None:
            chart['series_attr'][i]['format_xsrc'] = None
        else:
            if chart['series_attr'][i]['xsrc'] in list(field_name_types_dict.keys()):
                chart['series_attr'][i]['format_xsrc'] = field_name_types_dict[chart['series_attr'][i]['xsrc']]
            else:
                chart['series_attr'][i]['format_xsrc'] = "notfound"

        if chart['series_attr'][i]['ysrc'] == None:
            chart['series_attr'][i]['format_ysrc'] = None
        else:
            if chart['series_attr'][i]['ysrc'] in list(field_name_types_dict.keys()):
                chart['series_attr'][i]['format_ysrc'] = field_name_types_dict[chart['series_attr'][i]['ysrc']]
            else:
                chart['series_attr'][i]['format_ysrc'] = "notfound"
                
        if chart['series_attr'][i]['zsrc'] == None:
            chart['series_attr'][i]['format_zsrc'] = None
        else:
            if chart['series_attr'][i]['zsrc'] in list(field_name_types_dict.keys()):
                chart['series_attr'][i]['format_zsrc'] = field_name_types_dict[chart['series_attr'][i]['zsrc']]
            else:
                chart['series_attr'][i]['format_zsrc'] = "notfound"
        
    return table,chart


if __name__ == "__main__":
    table_codes = []
    chart_codes = []
    fids = []
    table_values = []
    origin_data_path = '../data/origin_data.tsv'
    raw_df_chunks = load_raw_data(fp = origin_data_path, chunk_size = CHUNK_SIZE)
    for i,chunk in enumerate(raw_df_chunks):
        chunk = clean_chunk(chunk)
        for chart_num, chart_obj in chunk.iterrows():
            fid = chart_obj['fid']
            chart_data = chart_obj['chart_data']
            table_data = chart_obj['table_data']
            layout = chart_obj['layout']

            chart = get_chart_code(fid, chart_data,layout)
            if chart == None:
                continue
            
            table, table_value = get_table_code(fid, table_data)
            field_name_types_dict = generate_field_types(table)
            table,chart = add_format(table,chart,field_name_types_dict)
            
            
            fids.append(fid)
            table_codes.append(table)
            chart_codes.append(chart)
            table_values.append(table_value)
           

    with open('processed_data/table_codes.pkl', 'wb') as f:
        pickle.dump(table_codes, f)
    with open('processed_data/chart_codes.pkl', 'wb') as f:
        pickle.dump(chart_codes, f)
    with open('processed_data/fids.pkl', 'wb') as f:
        pickle.dump(fids, f)
    with open('processed_data/table_values.pkl', 'wb') as f:
        pickle.dump(table_values, f)






