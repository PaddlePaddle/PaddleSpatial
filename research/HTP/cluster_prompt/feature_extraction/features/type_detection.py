# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
from scipy.stats import entropy, normaltest, mode
import pandas as pd

import re
import ast
from decimal import Decimal
from random import sample

from collections import OrderedDict

from datetime import datetime, date
import dateutil.parser as dparser
from .dateparser import DATE_FORMATS, is_date

general_types = ['c', 'q', 't']

# Type Detection Helpers
data_types = [
    'string',
    'integer',
    'decimal',
    'time'
]

data_type_to_general_type = {
    'string': 'c',
    'integer': 'q',
    'decimal': 'q',
    'time': 't'
}

integer_regex = re.compile(r"^-?(0|[1-9]\d*)(?<!-0)$")
decimal_regex = re.compile(r"^(?!-0?(\.0+)?$)-?(0|[1-9]\d*)?(\.\d+)?(?<=\d)$")


def detect_string(e):
    return True


def detect_integer(e):
    if e == '':
        return False
    try:
        if integer_regex.match(e):
            return True
    except BaseException:
        try:
            if float(e).is_integer():
                return True
        except BaseException:
            try:
                if float(locale.atoi(e)).is_integer():
                    return True
            except BaseException:
                pass
    return False


def detect_decimal(e):
    if decimal_regex.match(e):
        return True
    try:
        d = Decimal(e)
        return True
    except BaseException:
        try:
            value = locale.atof(e)
            if sys.version_info < (2, 7):
                value = str(e)
            return Decimal(e)
        except BaseException:
            pass
    return False


def detect_boolean(e):
    return


def detect_date(e):
    return


def detect_datetime(e):
    return


date_types = [datetime, date, np.datetime64]


def detect_time(e):
    if is_date(e):
        return True
    for date_type in date_types:
        if isinstance(e, date_type):
            return True
    return False


type_weights = {
    'integer': 4,
    'decimal': 3,
    'string': 1,
    'time': 2
}


def detect_field_type(v, num_samples=1000):
    type_scores = {
        'string': 1,
        'integer': 0,
        'decimal': 0,
        'time': 0
    }

    if len(v) > num_samples:
        v = sample(v, num_samples)
    for e in v:
        if e is None or e == '':
            continue
        e = str(e)
        if detect_string(e):
            type_scores['string'] += type_weights['string']
        if detect_integer(e):
            type_scores['integer'] += type_weights['integer']
        if detect_decimal(e):
            type_scores['decimal'] += type_weights['decimal']
        if detect_time(e):
            type_scores['time'] += type_weights['time']
    score_tuples = []
    for type_name, score in type_scores.items():
        score_tuples.append([type_name, score])
    final_field_type = max(score_tuples, key=lambda t: t[1])[0]
    return final_field_type, score_tuples
