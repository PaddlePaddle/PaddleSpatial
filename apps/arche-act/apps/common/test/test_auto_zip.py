# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import os
from apps.common.auto_zip import AutoZip

REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


def test_dict():
    """
    Desc:
        Test stub
    """
    path = os.path.join(REPO_ROOT, "apps/common/test/test_archive_1.zip")
    zp = AutoZip(path, ".txt")

    d = zp.as_dict()
    assert isinstance(d, dict)
    assert len(d) == 3

    d = zp.as_dict(include_zip_name=True)
    assert isinstance(d, dict)
    assert len(d) == 3
