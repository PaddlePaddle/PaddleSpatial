#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genregion",                     
    version="0.0.2",                        
    author="Ming Zhang & Yanyan Li",                  
    description="Region generation using urban road netwok",
    long_description=long_description,     
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    
    py_modules=["genregion"],           
    install_requires=['ordered-set', "shapely"]
)

