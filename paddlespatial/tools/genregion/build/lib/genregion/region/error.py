# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
import sys
import datetime

def debug(info):
    """Print the debug information.

    Args:
        info (str): A sentence that indicates a message for debugging.
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info)
    sys.stdout.flush()

class RegionError(Exception):
    """Region operation error.
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)

