# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from distutils.errors import PreprocessError
import re
from enum import Enum


class RoleType(Enum):
    """
    RoleType is a type of role.
    """
    ASSISTANT = "assistant"
    USER = "user"
    USER_PLAN = "user_plan"
    CRITIC = "critic"
    EMBODIMENT = "embodiment"
    DEFAULT = "default"


class ModelType(Enum):
    """
    ModelType is a type of model.
    """
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"
    GPT_4_32k = "gpt-4-32k"

    STUB = "stub"

    EB = "ernie-bot"
    EB_8K = "ernie-bot-turbo"
    EB_TURBO = "ernie-bot-speed"
    EB_4 = "ernie-bot-4"

    LLAMA_2 = "llama-2"
    VICUNA = "vicuna"
    VICUNA_16K = "vicuna-16k"

    @property
    def value_for_token(self) -> str:
        """
        Returns the value for Tokennizer.
        """
        return self.value if self.name != "STUB" else "gpt-3.5-turbo"
    
    @property
    def value_for_token_eb(self) -> str:
        """
        Returns the value for Ernie-Bot Tokennizer.
        """
        return self.value if self.name == "EB_4" else ""

    @property
    def api_for_eb_model(self) -> str:
        """
        Returns the api name for EB model.
        """
        if self is ModelType.EB:
            return "chat/compeletions"
        elif self is ModelType.EB_TURBO:
            # return "chat/eb-instant"
            return "chat/ernie_speed"
        elif self is ModelType.EB_8K:
            return "chat/ernie-4.0-turbo-8k"
        elif self is ModelType.EB_4:
            return "chat/completions_pro"
        else:
            raise ValueError("Unknown Ernie-Bot model type")

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model.

        Returns:
            bool: Whether this type of models belongs to OpenAI.
        """
        if self.name in {
                "GPT_3_5_TURBO",
                "GPT_3_5_TURBO_16K",
                "GPT_4",
                "GPT_4_32k",
        }:
            return True
        else:
            return False
    
    @property
    def is_ernie_bot(self) -> bool:
        r"""Returns whether this type of models is an ErnieBot-released model.

        Returns:
            bool: Whether this type of models belongs to ErnieBot.
        """
        if self.name in {
                "EB",
                "EB_TURBO",
                "EB_8K",
                "EB_4",
        }:
            return True
        else:
            return False

    @property
    def is_open_source(self) -> bool:
        r"""Returns whether this type of models is open-source.

        Returns:
            bool: Whether this type of models is open-source.
        """
        if self.name in {"LLAMA_2", "VICUNA", "VICUNA_16K"}:
            return True
        else:
            return False

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ModelType.GPT_3_5_TURBO:
            return 4096
        elif self is ModelType.GPT_3_5_TURBO_16K:
            return 16384
        elif self is ModelType.GPT_4:
            return 8192
        elif self is ModelType.GPT_4_32k:
            return 32768
        elif self is ModelType.STUB:
            return 4096
        elif self is ModelType.EB:
            return 4096
        elif self is ModelType.EB_TURBO:
            return 7168
        elif self is ModelType.EB_8K:
            return 8192
        elif self is ModelType.EB_4:
            return 5120
        elif self is ModelType.LLAMA_2:
            return 4096
        elif self is ModelType.VICUNA:
            # reference: https://lmsys.org/blog/2023-03-30-vicuna/
            return 2048
        elif self is ModelType.VICUNA_16K:
            return 16384
        else:
            raise ValueError("Unknown model type")

    def validate_model_name(self, model_name: str) -> bool:
        r"""Checks whether the model type and the model name matches.

        Args:
            model_name (str): The name of the model, e.g. "vicuna-7b-v1.5".
        Returns:
            bool: Whether the model type mathches the model name.
        """
        if self is ModelType.VICUNA:
            pattern = r'^vicuna-\d+b-v\d+\.\d+$'
            return bool(re.match(pattern, model_name))
        elif self is ModelType.VICUNA_16K:
            pattern = r'^vicuna-\d+b-v\d+\.\d+-16k$'
            return bool(re.match(pattern, model_name))
        elif self is ModelType.LLAMA_2:
            return (self.value in model_name.lower()
                    or "llama2" in model_name.lower())
        else:
            return self.value in model_name.lower()


class TaskType(Enum):
    """
    The type of tasks.
    """
    TRAVEL_ASSISTANT = "travel_assistant"

    CODE = "code"
    SOLUTION_EXTRACTION = "solution_extraction"
    
    DEFAULT = "default"


__all__ = ['RoleType', 'ModelType', 'TaskType']
