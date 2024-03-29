# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import apps.data_explorer.loader as loader


def test_load_datasets_smoke():
    """
    test_load_datasets_smoke
    """
    data = loader.load_datasets()
    assert data is not None
