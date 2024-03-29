# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Dict

from arche.models import (
    BaseModelBackend,
    OpenAIModel,
    OpenSourceModel,
    StubModel,
    ErnieBotModel
)
from arche.typing import ModelType


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType,
               model_config_dict: Dict) -> BaseModelBackend:
        r"""Creates an instance of `BaseModelBackend` of the specified type.

        Args:
            model_type (ModelType): Model for which a backend is created.
            model_config_dict (Dict): A dictionary that will be fed into
                the backend constructor.

        Raises:
            ValueError: If there is not backend for the model.

        Returns:
            BaseModelBackend: The initialized backend.
        """
        model_class: Any
        if model_type.is_openai:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        elif model_type.is_open_source:
            model_class = OpenSourceModel
        elif model_type.is_ernie_bot:
            model_class = ErnieBotModel
        else:
            raise ValueError(f"Unknown model type `{model_type}` is input")

        inst = model_class(model_type, model_config_dict)
        return inst
