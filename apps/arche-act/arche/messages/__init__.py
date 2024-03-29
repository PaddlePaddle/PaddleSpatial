# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Dict, Union


OpenAISystemMessage = Dict[str, str]
OpenAIAssistantMessage = Dict[str, str]
OpenAIUserMessage = Dict[str, str]
OpenAIChatMessage = Union[OpenAIUserMessage, OpenAIAssistantMessage]
OpenAIMessage = Union[OpenAISystemMessage, OpenAIChatMessage]
ErnieBotSystemMessage = Dict[str, str]
ErnieBotAssistantMessage = Dict[str, str]
ErnieBotUserMessage = Dict[str, str]
ErnieBotChatMessage = Union[ErnieBotUserMessage, ErnieBotAssistantMessage]
ErnieBotMessage = Union[ErnieBotSystemMessage, ErnieBotChatMessage]

from .base import BaseMessage  # noqa: E402
from .func_message import FunctionCallingMessage  # noqa: E402

__all__ = [
    'OpenAISystemMessage',
    'OpenAIAssistantMessage',
    'OpenAIUserMessage',
    'OpenAIChatMessage',
    'OpenAIMessage',
    'ErnieBotSystemMessage',
    'ErnieBotAssistantMessage',
    'ErnieBotUserMessage',
    'ErnieBotChatMessage',
    'ErnieBotMessage',
    'BaseMessage',
    'FunctionCallingMessage',
]
