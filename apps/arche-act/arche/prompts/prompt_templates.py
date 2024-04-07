# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import warnings
from typing import Any, Optional

from arche.prompts import TaskPromptTemplateDict, TextPrompt
from arche.typing import RoleType, TaskType


class PromptTemplateGenerator:
    r"""A class for generating prompt templates for tasks.

    Args:
        task_prompt_template_dict (TaskPromptTemplateDict, optional):
            A dictionary of task prompt templates for each task type. If not
            provided, an empty dictionary is used as default.
    """

    def __init__(
        self,
        task_prompt_template_dict: Optional[TaskPromptTemplateDict] = None,
    ) -> None:
        self.task_prompt_template_dict = (task_prompt_template_dict
                                          or TaskPromptTemplateDict())

    def get_prompt_from_key(self, task_type: TaskType, key: Any) -> TextPrompt:
        r"""Generates a text prompt using the specified :obj:`task_type` and
        :obj:`key`.

        Args:
            task_type (TaskType): The type of task.
            key (Any): The key used to generate the prompt.

        Returns:
            TextPrompt: The generated text prompt.

        Raises:
            KeyError: If failed to generate prompt using the specified
                :obj:`task_type` and :obj:`key`.
        """
        try:
            return self.task_prompt_template_dict[task_type][key]

        except KeyError:
            raise KeyError("Failed to get generate prompt template for "
                           f"task: {task_type.value} from key: {key}.")

    def get_system_prompt(
        self,
        task_type: TaskType,
        role_type: RoleType,
    ) -> TextPrompt:
        r"""Generates a text prompt for the system role, using the specified
        :obj:`task_type` and :obj:`role_type`.

        Args:
            task_type (TaskType): The type of task.
            role_type (RoleType): The type of role, either "USER" or
                "ASSISTANT".

        Returns:
            TextPrompt: The generated text prompt.

        Raises:
            KeyError: If failed to generate prompt using the specified
                :obj:`task_type` and :obj:`role_type`.
        """
        try:
            return self.get_prompt_from_key(task_type, role_type)

        except KeyError:
            prompt = "You are a helpful assistant."

            warnings.warn("Failed to get system prompt template for "
                          f"task: {task_type.value}, role: {role_type.value}. "
                          f"Set template to: {prompt}")

        return TextPrompt(prompt)

    def get_generate_tasks_prompt(
        self,
        task_type: TaskType,
    ) -> TextPrompt:
        r"""Gets the prompt for generating tasks for a given task type.

        Args:
            task_type (TaskType): The type of the task.

        Returns:
            TextPrompt: The generated prompt for generating tasks.
        """
        return self.get_prompt_from_key(task_type, "generate_tasks")

    def get_task_specify_prompt(
        self,
        task_type: TaskType,
    ) -> TextPrompt:
        r"""Gets the prompt for specifying a task for a given task type.

        Args:
            task_type (TaskType): The type of the task.

        Returns:
            TextPrompt: The generated prompt for specifying a task.
        """
        return self.get_prompt_from_key(task_type, "task_specify_prompt")

    def get_task_simplify_prompt(
        self,
        task_type: TaskType,
    ) -> TextPrompt:
        r"""Gets the prompt for simplifying a task for a given task type.

        Args:
            task_type (TaskType): The type of the task.

        Returns:
            TextPrompt: The generated prompt for simplifying a task.
        """
        return self.get_prompt_from_key(task_type, "task_simplify_prompt")
