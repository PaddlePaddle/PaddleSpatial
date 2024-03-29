# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from arche.messages import OpenAIAssistantMessage, OpenAIMessage

from .base import BaseMessage


@dataclass
class FunctionCallingMessage(BaseMessage):
    r"""Class for message objects used specifically for
    function-related messages.

    Args:
        func_name (Optional[str]): The name of the function used.
            (default: :obj:`None`)
        args (Optional[Dict]): The dictionary of arguments passed to the
            function. (default: :obj:`None`)
        result (Optional[Any]): The result of function execution.
            (default: :obj:`None`)
    """
    func_name: Optional[str] = None
    args: Optional[Dict] = None
    result: Optional[Any] = None

    def to_openai_message(self, role_at_backend: str) -> OpenAIMessage:
        r"""Converts the message to an :obj:`OpenAIMessage` object.

        Args:
            role_at_backend (str): The role of the message in OpenAI chat
                system, either :obj:`"system"`, :obj:`"user"`, or
                obj:`"assistant"`.

        Returns:
            OpenAIMessage: The converted :obj:`OpenAIMessage` object.
        """
        if role_at_backend not in {"assistant", "function"}:
            raise ValueError("Invalid role for function-related message: "
                             f"{role_at_backend}.")

        if role_at_backend == "assistant":
            return self.to_openai_assistant_message()
        else:
            return self.to_openai_function_message()

    def to_openai_assistant_message(self) -> OpenAIAssistantMessage:
        r"""Converts the message to an :obj:`OpenAIAssistantMessage` object.

        Returns:
            OpenAIAssistantMessage: The converted :obj:`OpenAIAssistantMessage`
                object.
        """
        if (not self.func_name) or (not self.args):
            raise ValueError(
                "Invalid request for converting into assistant message"
                " due to missing function name or arguments.")

        msg_dict: Dict[str, Any]
        msg_dict = {
            "role": "assistant",
            "content": self.content,
            "function_call": {
                "name": self.func_name,
                "arguments": str(self.args),
            }
        }

        return msg_dict

    def to_openai_function_message(self) -> OpenAIMessage:
        r"""Converts the message to an :obj:`OpenAIMessage` object
        with the role being "function".

        Returns:
            OpenAIMessage: The converted :obj:`OpenAIMessage` object
                with its role being "function".
        """
        if (not self.func_name) or (not self.result):
            raise ValueError(
                "Invalid request for converting into function message"
                " due to missing function name or results.")

        result_content = {"result": {str(self.result)}}
        msg_dict: Dict[str, str] = {
            "role": "function",
            "name": self.func_name,
            "content": f'{result_content}',
        }

        return msg_dict
