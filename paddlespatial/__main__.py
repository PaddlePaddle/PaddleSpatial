# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m paddlespatial方式直接执行。

Authors: zhoujingbo(zhoujingbo@baidu.com)
Date:    2021/10/19 10:30:45
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from paddlespatial.cmdline import main
sys.exit(main())
