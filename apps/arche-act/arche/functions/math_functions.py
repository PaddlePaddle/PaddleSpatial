# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import List

from .openai_function import OpenAIFunction


def add(a: int, b: int) -> int:
    r"""Adds two numbers.

    Args:
        a (integer): The first number to be added.
        b (integer): The second number to be added.

    Returns:
        integer: The sum of the two numbers.
    """
    return a + b


def sub(a: int, b: int) -> int:
    r"""Do subtraction between two numbers.

    Args:
        a (integer): The minuend in subtraction.
        b (integer): The subtrahend in subtraction.

    Returns:
        integer: The result of subtracting :obj:`b` from :obj:`a`.
    """
    return a - b


def mul(a: int, b: int) -> int:
    r"""Multiplies two integers.

    Args:
        a (integer): The multiplier in the multiplication.
        b (integer): The multiplicand in the multiplication.

    Returns:
        integer: The product of the two numbers.
    """
    return a * b


MATH_FUNCS: List[OpenAIFunction] = [
    OpenAIFunction(func) for func in [add, sub, mul]
]
