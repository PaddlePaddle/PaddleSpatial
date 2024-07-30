#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Extract field-level outcomes

E.g. whether a field is the single X axis, whether it is encoded as X or as Y
'''
from time import time, strftime
from collections import OrderedDict, Counter

from pprint import pprint

field_outcome_feature_names = [
    'fid',
    'field_id',
    'trace_type',
    'is_xsrc',
    'is_ysrc',
    'is_zsrc',
    'is_locationsrc',
    'is_labelsrc',
    'is_valuesrc',

    # The only src of that type
    'is_single_xsrc',
    'is_single_ysrc',
    'is_single_zsrc',
    'is_single_locationsrc',
    'is_single_labelsrc',
    'is_single_valuesrc',
    # 'is_x_axis_src',
    # 'is_y_axis_src',

    # Multiple occurrences
    'num_xsrc',
    'num_ysrc',
    'num_zsrc',
    'num_locationsrc',
    'num_labelsrc',
    'num_valuesrc',
]


def check_if_line_chart(d):
    if d.get('mode') in ['lines+markers', 'lines']:
        return True
    if d.get('line') and len(d.get('line').keys()) > 0:
        return True
    if d.get('marker') and 'line' in d.get('marker') and d.get('marker').get(
            'line') and d.get('marker').get('line').get('color') != 'transparent':
        return True
    return False


def format_field_id(s):
    return '{}:{}'.format(s.split(':')[0], s.split(':')[-1])


'''
Field ID = username + six letter hash (skipping middle dataset portion)
'''


def extract_field_outcomes(chart_obj):
    fid = chart_obj.fid
    user = fid.split(':')[0]
    chart_data = chart_obj.chart_data
    layout = chart_obj.layout
    table_data = chart_obj.table_data

    field_outcomes = {}

    fields_by_id = {}
    fields = table_data[list(table_data.keys())[0]]['cols']
    for field_name, d in fields.items():
        field_id = '{}:{}'.format(user, d['uid'])
        fields_by_id[d['uid']] = d
        field_outcomes[field_id] = OrderedDict(
            [(f, None) for f in field_outcome_feature_names])
        field_outcomes[field_id]['fid'] = fid
        field_outcomes[field_id]['field_id'] = field_id

    try:
        # print('Field IDs:', field_outcomes.keys())

        # Outcomes
        num_traces = 0  # Not counting axes
        trace_types = []

        xsrcs = []
        ysrcs = []
        zsrcs = []
        locationsrcs = []
        labelsrcs = []
        valuesrcs = []
        srcs = []
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
            zsrc = d.get('zsrc')
            locationsrc = d.get('locationsrc')
            labelsrc = d.get('labelsrc')
            valuesrc = d.get('valuesrc')

            if xsrc:
                field_id = format_field_id(xsrc)
                srcs.append(field_id)
                xsrcs.append(field_id)
            if ysrc:
                field_id = format_field_id(ysrc)
                srcs.append(field_id)
                ysrcs.append(field_id)
            if zsrc:
                field_id = format_field_id(zsrc)
                srcs.append(field_id)
                zsrcs.append(field_id)
            if locationsrc:
                field_id = format_field_id(locationsrc)
                srcs.append(field_id)
                locationsrcs.append(field_id)
            if labelsrc:
                field_id = format_field_id(labelsrc)
                srcs.append(field_id)
                labelsrcs.append(field_id)
            if valuesrc:
                field_id = format_field_id(valuesrc)
                srcs.append(field_id)
                valuesrcs.append(field_id)

            # Add trace types
            unique_srcs = set([s for s in srcs if s])
            for field_id in unique_srcs:
                field_outcomes[field_id]['trace_type'] = t

            # Add axis data
            if d.get('xaxis'):
                x_axes.append(d.get('xaxis'))
            if d.get('yaxis'):
                y_axes.append(d.get('yaxis'))
            if d.get('xaxis') or d.get('yaxis'):
                axes.append([d.get('xaxis'), d.get('yaxis')])

        if xsrcs:
            if len(xsrcs) > 1 and len(set(xsrcs)) == 1:
                field_outcomes[xsrcs[0]]['is_single_xsrc'] = True
            for field_id, count in Counter(xsrcs).items():
                field_outcomes[field_id]['is_xsrc'] = True
                field_outcomes[field_id]['num_xsrc'] = count
        if ysrcs:
            if len(ysrcs) > 1 and len(set(ysrcs)) == 1:
                field_outcomes[ysrcs[0]]['is_single_ysrc'] = True
            for field_id, count in Counter(ysrcs).items():
                field_outcomes[field_id]['is_ysrc'] = True
                field_outcomes[field_id]['num_ysrc'] = count
        if zsrcs:
            if len(zsrcs) > 1 and len(set(zsrcs)) == 1:
                field_outcomes[zsrcs[0]]['is_single_zsrc'] = True
            for field_id, count in Counter(zsrc).items():
                field_outcomes[field_id]['is_zsrc'] = True
                field_outcomes[field_id]['num_zsrc'] = count
        if locationsrcs:
            if len(locationsrcs) > 1 and len(set(locationsrcs)) == 1:
                field_outcomes[locationsrcs[0]]['is_single_locationsrc'] = True
            for field_id, count in Counter(locationsrcs).items():
                field_outcomes[field_id]['is_locationsrc'] = True
                field_outcomes[field_id]['num_locationsrc'] = count
        if labelsrcs:
            if len(labelsrcs) > 1 and len(set(labelsrcs)) == 1:
                field_outcomes[labelsrcs[0]]['is_single_labelsrc'] = True
            for field_id, count in Counter(labelsrcs).items():
                field_outcomes[field_id]['is_labelsrc'] = True
                field_outcomes[field_id]['num_labelsrc'] = count
        if valuesrcs:
            if len(valuesrcs) > 1 and len(set(valuesrcs)) == 1:
                field_outcomes[valuesrcs[0]]['is_single_valuesrc'] = True
            for field_id, count in Counter(valuesrcs).items():
                field_outcomes[field_id]['is_valuesrc'] = True
                field_outcomes[field_id]['num_valuesrc'] = count

        # ################
        # # Axes
        # ################
        x_layout_axes = []
        y_layout_axes = []

        for k in layout.keys():
            if 'xaxis' in k:
                x_layout_axes.append(k)
            if 'yaxis' in k:
                y_layout_axes.append(k)

    except Exception as e:
        return field_outcomes.values()

    return field_outcomes.values()
