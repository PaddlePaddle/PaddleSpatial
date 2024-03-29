# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import os
import urllib.request

import gradio as gr

from apps.data_explorer.data_explorer import construct_blocks, parse_arguments
from apps.data_explorer.loader import REPO_ROOT


def test_app():
    """
    Test the app.
    """
    test_data_url = ("https://storage.googleapis.com/"
                     "camel-bucket/datasets/test/DATA.zip")
    data_dir = os.path.join(REPO_ROOT, "datasets_test")
    test_file_path = os.path.join(data_dir, os.path.split(test_data_url)[1])
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(test_data_url, test_file_path)

    blocks = construct_blocks(data_dir, None)

    assert isinstance(blocks, gr.Blocks)


def test_utils():
    """
    Test the utils.
    """
    args = parse_arguments()
    assert args is not None
