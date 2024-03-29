# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from arche.messages import (
    OpenAIAssistantMessage,
    OpenAIMessage,
    OpenAISystemMessage,
    OpenAIUserMessage,
    ErnieBotAssistantMessage,
    ErnieBotMessage,
    ErnieBotSystemMessage,
    ErnieBotUserMessage,
)
from arche.prompts import CodePrompt, TextPrompt
from arche.typing import RoleType


@dataclass
class BaseMessage:
    r"""Base class for message objects used in multi-agent chat system.

    Args:
        role_name (str): The name of the user or assistant role.
        role_type (RoleType): The type of role, either
            :obj:`RoleType.ASSISTANT` or :obj:`RoleType.USER`.
        meta_dict (Optional[Dict[str, str]]): Additional metadata dictionary
            for the message.
        role (str): The role of the message in OpenAI chat system, either
            :obj:`"system"`, :obj:`"user"`, or :obj:`"assistant"`.
        content (str): The content of the message.
    """
    role_name: str
    role_type: RoleType
    meta_dict: Optional[Dict[str, str]]
    content: str

    @classmethod
    def make_user_message(
            cls, role_name: str, content: str,
            meta_dict: Optional[Dict[str, str]] = None) -> 'BaseMessage':
        """
        Desc:
            Construbt a user message
        """
        return cls(role_name, RoleType.USER, meta_dict, content)
    
    @classmethod
    def make_assistant_message(
            cls, role_name: str, content: str,
            meta_dict: Optional[Dict[str, str]] = None) -> 'BaseMessage':
        """
        Desc:
            Construbt an assistant message
        """
        return cls(role_name, RoleType.ASSISTANT, meta_dict, content)

    def create_new_instance(self, content: str) -> "BaseMessage":
        r"""Create a new instance of the :obj:`BaseMessage` with updated
        content.

        Args:
            content (str): The new content value.

        Returns:
            BaseMessage: The new instance of :obj:`BaseMessage`.
        """
        return self.__class__(role_name=self.role_name,
                              role_type=self.role_type,
                              meta_dict=self.meta_dict, content=content)

    def __add__(self, other: Any) -> Union["BaseMessage", Any]:
        r"""Addition operator override for :obj:`BaseMessage`.

        Args:
            other (Any): The value to be added with.

        Returns:
            Union[BaseMessage, Any]: The result of the addition.
        """
        if isinstance(other, BaseMessage):
            combined_content = self.content.__add__(other.content)
        elif isinstance(other, str):
            combined_content = self.content.__add__(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self)}' and "
                f"'{type(other)}'")
        return self.create_new_instance(combined_content)

    def __mul__(self, other: Any) -> Union["BaseMessage", Any]:
        r"""Multiplication operator override for :obj:`BaseMessage`.

        Args:
            other (Any): The value to be multiplied with.

        Returns:
            Union[BaseMessage, Any]: The result of the multiplication.
        """
        if isinstance(other, int):
            multiplied_content = self.content.__mul__(other)
            return self.create_new_instance(multiplied_content)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(self)}' and "
                f"'{type(other)}'")

    def __len__(self) -> int:
        r"""Length operator override for :obj:`BaseMessage`.

        Returns:
            int: The length of the content.
        """
        return len(self.content)

    def __contains__(self, item: str) -> bool:
        r"""Contains operator override for :obj:`BaseMessage`.

        Args:
            item (str): The item to check for containment.

        Returns:
            bool: :obj:`True` if the item is contained in the content,
                :obj:`False` otherwise.
        """
        return item in self.content

    def extract_text_and_code_prompts(
            self) -> Tuple[List[TextPrompt], List[CodePrompt]]:
        r"""Extract text and code prompts from the message content.

        Returns:
            Tuple[List[TextPrompt], List[CodePrompt]]: A tuple containing a
                list of text prompts and a list of code prompts extracted
                from the content.
        """
        text_prompts: List[TextPrompt] = []
        code_prompts: List[CodePrompt] = []

        lines = self.content.split("\n")
        idx = 0
        start_idx = 0
        while idx < len(lines):
            while idx < len(lines) and (
                    not lines[idx].lstrip().startswith("```")):
                idx += 1
            text = "\n".join(lines[start_idx:idx]).strip()
            text_prompts.append(TextPrompt(text))

            if idx >= len(lines):
                break

            code_type = lines[idx].strip()[3:].strip()
            idx += 1
            start_idx = idx
            while not lines[idx].lstrip().startswith("```"):
                idx += 1
            code = "\n".join(lines[start_idx:idx]).strip()
            code_prompts.append(CodePrompt(code, code_type=code_type))

            idx += 1
            start_idx = idx

        return text_prompts, code_prompts

    def to_openai_message(self, role_at_backend: str) -> OpenAIMessage:
        r"""Converts the message to an :obj:`OpenAIMessage` object.

        Args:
            role_at_backend (str): The role of the message in OpenAI chat
                system, either :obj:`"system"`, :obj:`"user"`, or
                obj:`"assistant"`.

        Returns:
            OpenAIMessage: The converted :obj:`OpenAIMessage` object.
        """
        if role_at_backend not in {"system", "user", "assistant"}:
            raise ValueError(f"Unrecognized role: {role_at_backend}")
        return {"role": role_at_backend, "content": self.content}

    def to_openai_system_message(self) -> OpenAISystemMessage:
        r"""Converts the message to an :obj:`OpenAISystemMessage` object.

        Returns:
            OpenAISystemMessage: The converted :obj:`OpenAISystemMessage`
                object.
        """
        return {"role": "system", "content": self.content}

    def to_openai_user_message(self) -> OpenAIUserMessage:
        r"""Converts the message to an :obj:`OpenAIUserMessage` object.

        Returns:
            OpenAIUserMessage: The converted :obj:`OpenAIUserMessage` object.
        """
        return {"role": "user", "content": self.content}

    def to_openai_assistant_message(self) -> OpenAIAssistantMessage:
        r"""Converts the message to an :obj:`OpenAIAssistantMessage` object.

        Returns:
            OpenAIAssistantMessage: The converted :obj:`OpenAIAssistantMessage`
                object.
        """
        return {"role": "assistant", "content": self.content}

    def to_eb_message(self, role_at_backend: str) -> ErnieBotMessage:
        r"""Converts the message to an :obj:`ErnieBotMessage` object.

        Args:
            role_at_backend (str): The role of the message in ErnieBot chat
                system, either :obj:`"system"`, :obj:`"user"`, or
                obj:`"assistant"`.

        Returns:
            ErnieBotMessage: The converted :obj:`ErnieBotMessage` object.
        """
        if role_at_backend not in {"system", "user", "assistant"}:
            raise ValueError(f"Unrecognized role: {role_at_backend}")
        return {"role": role_at_backend, "content": self.content}

    def to_eb_system_message(self) -> ErnieBotSystemMessage:
        r"""Converts the message to an :obj:`ErnieBotSystemMessage` object.

        Returns:
            ErnieBotSystemMessage: The converted :obj:`ErnieBotSystemMessage`
                object.
        """
        return {"role": "system", "content": self.content}

    def to_eb_user_message(self) -> ErnieBotUserMessage:
        r"""Converts the message to an :obj:`ErnieBotUserMessage` object.

        Returns:
            ErnieBotUserMessage: The converted :obj:`ErnieBotUserMessage` object.
        """
        return {"role": "user", "content": self.content}

    def to_eb_assistant_message(self) -> ErnieBotAssistantMessage:
        r"""Converts the message to an :obj:`ErnieBotAssistantMessage` object.

        Returns:
            ErnieBotAssistantMessage: The converted :obj:`ErnieBotAssistantMessage`
                object.
        """
        return {"role": "assistant", "content": self.content}

    def to_dict(self) -> Dict:
        r"""Converts the message to a dictionary.

        Returns:
            dict: The converted dictionary.
        """
        return {
            "role_name": self.role_name,
            "role_type": self.role_type.name,
            **(self.meta_dict or {}),
            "content": self.content,
        }
