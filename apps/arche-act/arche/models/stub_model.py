# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Dict, List, Optional

from arche.messages import OpenAIMessage
from arche.models import BaseModelBackend
from arche.typing import ModelType
from arche.utils import BaseTokenCounter


class StubTokenCounter(BaseTokenCounter):
    """
    A dummy token counter for STUB models.
    """
    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        r"""Token counting for STUB models, directly returning a constant.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            int: A constant to act as the number of the tokens in the
                messages.
        """
        return 10


class StubModel(BaseModelBackend):
    r"""A dummy model used for unit tests."""
    model_type = ModelType.STUB

    def __init__(self, model_type: ModelType,
                 model_config_dict: Dict[str, Any]) -> None:
        r"""All arguments are unused for the dummy model."""
        super().__init__(model_type, model_config_dict)
        self._token_counter: Optional[BaseTokenCounter] = None

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = StubTokenCounter()
        return self._token_counter

    def run(self, messages: List[Dict]) -> Dict[str, Any]:
        r"""Run fake inference by returning a fixed string.
        All arguments are unused for the dummy model.

        Returns:
            Dict[str, Any]: Response in the OpenAI API format.
        """
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )

    def check_model_config(self):
        r"""Directly pass the check on arguments to STUB model.
        """
        pass
