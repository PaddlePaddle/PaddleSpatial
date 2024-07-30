#!/usr/bin/python
# -*- coding: utf-8 -*-

from time import time, strftime
from collections import OrderedDict, Counter

from pprint import pprint

outcome_properties = [
    'fid',

    # Subplots
    'both_single_srcs',
    'has_single_src',
    'has_single_x_src',
    'has_single_y_src',
    'single_x_src_id',
    'single_y_src_id',
    'single_x_src_order',
    'single_y_src_order',

    # Axes
    'num_x_axes',
    'num_y_axes',
    'num_unique_x_axes',
    'num_unique_y_axes',
    'one_subplot',

    # Traces
    'num_traces',
    'num_trace_types',
    'trace_types',

    # Chart-level
    'is_all_one_trace_type',
    'all_one_trace_type',

    # Etc
    'num_fields_used_by_data',

    # 'chart_data',
    # 'layout'
]


def check_if_line_chart(d):
    if d.get('mode') in ['lines+markers', 'lines']:
        return True
    if d.get('line') and len(d.get('line').keys()) > 0:
        return True
    if d.get('marker') and 'line' in d.get('marker'):
        return True

    return False


def extract_chart_outcomes(chart_obj):
    fid = chart_obj.fid
    chart_data = chart_obj.chart_data
    layout = chart_obj.layout
    table_data = chart_obj.table_data
    outcomes = OrderedDict([(f, None) for f in outcome_properties])

    fields_by_id = {}
    fields = table_data[list(table_data.keys())[0]]['cols']
    for field_name, d in fields.items():
        fields_by_id[d['uid']] = d

    # Outcomes
    outcomes['fid'] = fid
    num_traces = 0  # Not counting axes
    trace_types = []

    x_srcs = []
    y_srcs = []
    x_axes = []
    y_axes = []
    axes = []
    # Parse Chart Data
    for d in chart_data:
        t = d.get('type')
        if t:
            if t == 'scatter' and check_if_line_chart(d):
                t = 'line'
            num_traces += 1
            trace_types.append(t)

        xsrc = d.get('xsrc')
        ysrc = d.get('ysrc')
        if xsrc:
            x_srcs.append(xsrc)
        if ysrc:
            y_srcs.append(ysrc)

        if d.get('xaxis'):
            x_axes.append(d.get('xaxis'))
        if d.get('yaxis'):
            y_axes.append(d.get('yaxis'))
        if d.get('xaxis') or d.get('yaxis'):
            axes.append([d.get('xaxis'), d.get('yaxis')])

    num_fields_used_by_data = len(set(x_srcs)) + len(set(y_srcs))

    ################
    # Trace Data Sources
    ################
    both_single_srcs = False
    has_single_src = False
    has_single_x_src = False
    has_single_y_src = False
    single_x_src_id = None
    single_y_src_id = None
    single_x_src_order = None
    single_y_src_order = None
    # single_axis_src_order = None
    if (len(x_srcs) > 1 and len(set(x_srcs)) == 1):
        has_single_src = True
        has_single_x_src = True
        single_x_src_id = list(set(x_srcs))[0]
        single_x_src_order = fields_by_id[single_x_src_id.split(
            ':')[-1]]['order']

    if (len(y_srcs) > 1 and len(set(y_srcs)) == 1):
        has_single_src = True
        has_single_y_src = True
        single_y_src_id = list(set(y_srcs))[0]
        single_y_src_order = fields_by_id[single_y_src_id.split(
            ':')[-1]]['order']

    if has_single_x_src and has_single_y_src:
        both_single_srcs = True

    ################
    # Axes
    ################
    x_layout_axes = []
    y_layout_axes = []
    # {u'domain': [0, 0.99], u'showticklabels': True, u'autorange': True, u'ticks': u'outside', u'showgrid': True, u'range': [1345815520485.415, 1442917279514.585], u'gridcolor': u'rgb(255,255,255)', u'tickcolor': u'rgb(127,127,127)', u'zeroline': False, u'showline': False, u'type': u'date', u'anchor': u'y'}
    for k in layout.keys():
        if 'xaxis' in k:
            x_layout_axes.append(k)
        if 'yaxis' in k:
            y_layout_axes.append(k)

    unique_x_axes = set(x_axes)
    unique_y_axes = set(y_axes)
    one_subplot = False
    if (len(unique_x_axes) <= 1) and (len(unique_y_axes) <= 1):
        one_subplot = True

    ################
    # Trace Types
    ################
    all_one_trace_type = None
    is_all_one_trace_type = False
    if len(set(trace_types)) == 1:
        all_one_trace_type = trace_types[0]
        is_all_one_trace_type = True

    ################
    # Populating Final Results
    ################
    outcomes['both_single_srcs'] = both_single_srcs
    outcomes['has_single_src'] = has_single_src
    outcomes['has_single_x_src'] = has_single_x_src
    outcomes['has_single_y_src'] = has_single_y_src
    outcomes['single_x_src_id'] = single_x_src_id
    outcomes['single_y_src_id'] = single_y_src_id
    outcomes['single_x_src_order'] = single_x_src_order
    outcomes['single_y_src_order'] = single_y_src_order

    outcomes['num_x_axes'] = len(x_layout_axes)
    outcomes['num_y_axes'] = len(y_layout_axes)
    outcomes['num_unique_x_axes'] = len(set(x_layout_axes))
    outcomes['num_unique_y_axes'] = len((y_layout_axes))
    outcomes['one_subplot'] = one_subplot

    outcomes['num_traces'] = len(trace_types)
    outcomes['num_trace_types'] = len(set(trace_types))
    outcomes['trace_types'] = trace_types
    outcomes['is_all_one_trace_type'] = is_all_one_trace_type
    outcomes['all_one_trace_type'] = all_one_trace_type

    outcomes['num_fields_used_by_data'] = num_fields_used_by_data

    # outcomes['chart_data'] = chart_data
    # outcomes['layout'] = layout

    return outcomes
