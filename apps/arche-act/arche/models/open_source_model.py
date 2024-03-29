# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from types import GeneratorType
from typing import Any, Dict, List, Optional

from arche.configs import OPENAI_API_PARAMS
from arche.messages import OpenAIMessage
from arche.models import BaseModelBackend
from arche.typing import ModelType
from arche.utils import BaseTokenCounter, OpenSourceTokenCounter


class OpenSourceModel(BaseModelBackend):
    r"""Class for interace with OpenAI-API-compatible servers running
    open-source models.
    """

    def __init__(
        self,
        model_type: ModelType,
        model_config_dict: Dict[str, Any],
    ) -> None:
        r"""Constructor for model backends of Open-source models.

        Args:
            model_type (ModelType): Model for which a backend is created.
            model_config_dict (Dict[str, Any]): A dictionary that will
                be fed into :obj:`openai.ChatCompletion.create()`.
        """
        super().__init__(model_type, model_config_dict)
        self._token_counter: Optional[BaseTokenCounter] = None

        # Check whether the input model type is open-source
        if not model_type.is_open_source:
            raise ValueError(
                f"Model `{model_type}` is not a supported open-source model.")

        # Check whether input model path is empty
        model_path: Optional[str] = (self.model_config_dict.get(
            "model_path", None))
        if not model_path:
            raise ValueError("Path to open-source model is not provided.")
        self.model_path: str = model_path

        # Check whether the model name matches the model type
        self.model_name: str = self.model_path.split('/')[-1]
        if not self.model_type.validate_model_name(self.model_name):
            raise ValueError(
                f"Model name `{self.model_name}` does not match model type "
                f"`{self.model_type.value}`.")

        # Load the server URL and check whether it is None
        server_url: Optional[str] = (self.model_config_dict.get(
            "server_url", None))
        if not server_url:
            raise ValueError(
                "URL to server running open-source LLM is not provided.")
        self.server_url: str = server_url

        # Replace `model_config_dict` with only the params to be
        # passed to OpenAI API
        self.model_config_dict = self.model_config_dict["api_params"].__dict__

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenSourceTokenCounter(
                self.model_type, self.model_path)
        return self._token_counter

    def run(
        self,
        messages: List[Dict],
    ) -> Dict[str, Any]:
        r"""Run inference of OpenAI-API-style chat completion.

        Args:
            messages (List[Dict]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Dict[str, Any]: Response in the OpenAI API format.
        """
        import openai
        openai.api_base = self.server_url

        messages_openai: List[OpenAIMessage] = messages
        response = openai.ChatCompletion.create(messages=messages_openai,
                                                model=self.model_name,
                                                **self.model_config_dict)
        if not self.stream:
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected batch return from OpenAI API")
        else:
            if not isinstance(response, GeneratorType):
                raise RuntimeError("Unexpected stream return from OpenAI API")
        return response

    def check_model_config(self):
        r"""Check whether the model configuration is valid for open-source
        model backends.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to OpenAI API, or it does not contain
                :obj:`model_path` or :obj:`server_url`.
        """
        if ("model_path" not in self.model_config_dict
                or "server_url" not in self.model_config_dict):
            raise ValueError(
                "Invalid configuration for open-source model backend with "
                ":obj:`model_path` or :obj:`server_url` missing.")

        for param in self.model_config_dict["api_params"].__dict__:
            if param not in OPENAI_API_PARAMS:
                raise ValueError(f"Unexpected argument `{param}` is "
                                 "input into open-source model backend.")

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode,
            which sends partial results each time.
        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
