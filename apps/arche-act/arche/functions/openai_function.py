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


class OpenAIFunction:
    r"""An abstraction of a function that OpenAI chat models can call. See
    https://platform.openai.com/docs/guides/gpt/function-calling. If
    :obj:`description` and :obj:`parameters` are both :obj:`None`, try to use
    document parser to generate them.

    # flake8: noqa :E501
    Args:
        func (Callable): The function to call.
        name (str, optional): The name of the function to be called. Must be
            a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum
            length of 64. If :obj:`None`, use the name of :obj:`func`. 
            (default: :obj:`None`)
        description (str, optional): The description of what the
            function does. (default: :obj:`None`)
        parameters (dict, optional): The parameters the
            functions accepts, described as a JSON Schema object. See the
            `Function calling guide <https://platform.openai.com/docs/guides/gpt/function-calling>`_
            for examples, and the `JSON Schema reference <https://json-schema.org/understanding-json-schema/>`_
            for documentation about the format.
    """

    def __init__(self, func: Callable, name: Optional[str] = None,
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
