# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Callable, Dict, Optional

from jsonschema.validators import Draft202012Validator as JSONValidator

from arche.utils import parse_doc


class ErnieBotFunction:
    r"""An abstraction of a function that ErnieBot chat models can call. See
    https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11. If
    :obj:`description` and :obj:`parameters` are both :obj:`None`, try to use
    document parser to generate them.
    """

    def __init__(
            self, func: Callable, 
            name: Optional[str] = None,
            description: Optional[str] = None,
            parameters: Optional[Dict[str, Any]] = None):
        self.func = func
        self.name = name or func.__name__

        info = parse_doc(self.func)
        self.description = description or info["description"]
        self.parameters = parameters or info["parameters"]

    @property
    def parameters(self) -> Dict[str, Any]:
        r"""Getter method for the property :obj:`parameters`.

        Returns:
            Dict[str, Any]: the dictionary containing information of
                parameters of this function.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dict[str, Any]):
        r"""Setter method for the property :obj:`parameters`. It will
        firstly check if the input parameters schema is valid. If invalid,
        the method will raise :obj:`jsonschema.exceptions.SchemaError`.

        Args:
            value (Dict[str, Any]): the new dictionary value for the
                function's parameters.
        """
        JSONValidator.check_schema(value)
        self._parameters = value

    def as_dict(self) -> Dict[str, Any]:
        r"""Method to represent the information of this function into
        a dictionary object.

        Returns:
            Dict[str, Any]: The dictionary object containing information
                of this function's name, description and parameters.
        """
        return {
            attr: getattr(self, attr)
            for attr in ["name", "description", "parameters"]
            if getattr(self, attr) is not None
        }
