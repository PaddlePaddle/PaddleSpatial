# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Dict, Generator, List, Optional, Set, Tuple
from loguru import logger

from arche.messages import BaseMessage
from arche.prompts import PromptTemplateGenerator, TextPrompt
from arche.typing import RoleType, TaskType


class SystemMessageGenerator:
    r"""System message generator for agents.

    Args:
        task_type (TaskType, optional): The task type.
            (default: :obj:`TaskType.AI_SOCIETY`)
        sys_prompts (Optional[Dict[RoleType, str]], optional): The prompts of
            the system messages for each role type. (default: :obj:`None`)
        sys_msg_meta_dict_keys (Optional[Set[str]], optional): The set of keys
            of the meta dictionary used to fill the prompts.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.TRAVEL_ASSISTANT,
        sys_prompts: Optional[Dict[RoleType, str]] = None,
        sys_msg_meta_dict_keys: Optional[Set[str]] = None,
    ) -> None:
        self.sys_prompts: Dict[RoleType, str]

        if sys_prompts is not None:
            self.sys_prompts = sys_prompts
            self.sys_msg_meta_dict_keys = sys_msg_meta_dict_keys or set()
        else:
            assistant_prompt_template = PromptTemplateGenerator(
            ).get_system_prompt(
                task_type,
                RoleType.ASSISTANT,
            )
            user_prompt_template = PromptTemplateGenerator(
            ).get_system_prompt(
                task_type,
                RoleType.USER,
            )
            user_plan_prompt_template = PromptTemplateGenerator(
            ).get_system_prompt(
                task_type,
                RoleType.USER_PLAN
            )
            critic_prompt_template = PromptTemplateGenerator(
            ).get_system_prompt(
                task_type,
                RoleType.CRITIC,
            )
            embodiment_prompt_template = PromptTemplateGenerator(
            ).get_system_prompt(
                task_type,
                RoleType.EMBODIMENT,
            )

            self.sys_prompts = dict()
            self.sys_prompts[RoleType.ASSISTANT] = assistant_prompt_template
            self.sys_prompts[RoleType.USER] = user_prompt_template
            self.sys_prompts[RoleType.CRITIC] = critic_prompt_template
            self.sys_prompts[RoleType.EMBODIMENT] = embodiment_prompt_template
            self.sys_prompts[RoleType.USER_PLAN] = user_plan_prompt_template

            self.sys_msg_meta_dict_keys = (
                assistant_prompt_template.key_words
                | user_prompt_template.key_words
                | critic_prompt_template.key_words
                | embodiment_prompt_template.key_words
                | user_plan_prompt_template.key_words)

        if RoleType.DEFAULT not in self.sys_prompts:
            self.sys_prompts[RoleType.DEFAULT] = "You are a helpful assistant."

    def validate_meta_dict_keys(self, meta_dict: Dict[str, str]) -> None:
        r"""Validates the keys of the meta_dict.

        Args:
            meta_dict (Dict[str, str]): The dictionary to validate.
        """
        if not set(meta_dict.keys()).issubset(self.sys_msg_meta_dict_keys):
            raise ValueError("The keys of the meta_dict should be in "
                             f"{self.sys_msg_meta_dict_keys}. "
                             f"Got {set(meta_dict.keys())} instead.")

    def from_dict(
        self,
        meta_dict: Dict[str, str],
        role_tuple: Tuple[str, RoleType] = ("", RoleType.DEFAULT),
    ) -> BaseMessage:
        r"""Generates a system message from a dictionary.

        Args:
            meta_dict (Dict[str, str]): The dictionary containing the
                information to generate the system message.
            role_tuple (Tuple[str, RoleType], optional): The tuple containing
                the role name and role type. (default: ("", RoleType.DEFAULT))

        Returns:
            BaseMessage: The generated system message.
        """
        self.validate_meta_dict_keys(meta_dict)
        role_name, role_type = role_tuple
        sys_prompt = self.sys_prompts[role_type]
        sys_prompt = sys_prompt.format(**meta_dict)
        logger.debug(f"meta_dict: {meta_dict}")
        logger.debug(f"role_tuple: {role_tuple}, sys_prompt: {sys_prompt}")

        return BaseMessage(role_name=role_name, role_type=role_type,
                           meta_dict=meta_dict, content=sys_prompt)

    def from_dicts(
        self,
        meta_dicts: List[Dict[str, str]],
        role_tuples: List[Tuple[str, RoleType]],
    ) -> List[BaseMessage]:
        r"""Generates a list of system messages from a list of dictionaries.

        Args:
            meta_dicts (List[Dict[str, str]]): A list of dictionaries
                containing the information to generate the system messages.
            role_tuples (List[Tuple[str, RoleType]]): A list of tuples
                containing the role name and role type for each system message.

        Returns:
            List[BaseMessage]: A list of generated system messages.

        Raises:
            ValueError: If the number of meta_dicts and role_tuples are
                different.
        """
        if len(meta_dicts) != len(role_tuples):
            raise ValueError(
                "The number of meta_dicts and role_types should be the same.")

        return [
            self.from_dict(meta_dict, role_tuple)
            for meta_dict, role_tuple in zip(meta_dicts, role_tuples)
        ]


class RoleNameGenerator:
    """
    A generator that generates role names.
    """
    def __init__(self, assistant_role_names_path:
                 str = "data/ai_society/assistant_roles.txt",
                 user_role_names_path: str = "data/ai_society/user_roles.txt",
                 assistant_role_names: Optional[List[str]] = None,
                 user_role_names: Optional[List[str]] = None) -> None:

        if assistant_role_names is None:
            with open(assistant_role_names_path, "r") as f:
                assistant_role_names_: List[str] = f.read().splitlines()
                self.assistant_role_names = [
                    " ".join(name.split(" ")[1:])
                    for name in assistant_role_names_
                ]
        else:
            self.assistant_role_names = assistant_role_names

        if user_role_names is None:
            with open(user_role_names_path, "r") as f:
                user_role_names_: List[str] = f.read().splitlines()
                self.user_role_names = [
                    " ".join(name.split(" ")[1:]) for name in user_role_names_
                ]
        else:
            self.user_role_names = user_role_names

    def from_role_files(self) -> Generator[Tuple, None, None]:
        """
        Returns a generator that yields tuples of assistant role names and user
        """
        for assistant_role_name in self.assistant_role_names:
            for user_role_name in self.user_role_names:
                yield (assistant_role_name, user_role_name)


class SingleTxtGenerator:
    """
    A generator that generates data from a single text file.
    """
    def __init__(
        self,
        text_file_path: str,
    ) -> None:

        with open(text_file_path, "r") as f:
            data_list: List[str] = f.read().splitlines()
            self.data_list = [
                " ".join(name.split(" ")[1:]) for name in data_list
            ]

    def from_role_files(self) -> Generator[str, None, None]:
        """
        Generate data from a single text file.
        """
        for data in self.data_list:
            yield data


class CodeTaskPromptGenerator:
    """
    A generator that generates task prompts for code.
    """
    def __init__(
        self,
        num_tasks: int = 50,
    ) -> None:

        self.generate_tasks_prompt = PromptTemplateGenerator(
        ).get_generate_tasks_prompt(TaskType.CODE)

        self.num_tasks = num_tasks

    def from_role_files(
        self, languages_path: str = "data/code/languages.txt",
        domains_path: str = "data/code/domains.txt"
    ) -> Generator[Tuple[TextPrompt, str, str], None, None]:
        """
        Generate task prompts from languages and domains.
        """
        language_generator = SingleTxtGenerator(
            languages_path).from_role_files()

        for language in language_generator:
            domains_generator = SingleTxtGenerator(
                domains_path).from_role_files()
            for domain in domains_generator:
                generated_tasks_prompt = self.generate_tasks_prompt.format(
                    language=language, domain=domain, num_tasks=self.num_tasks)
                yield generated_tasks_prompt, language, domain

    def from_role_generator(
        self, role_generator: Generator[Tuple, None, None]
    ) -> Generator[str, None, None]:
        """
        Generate task prompts from a role generator.
        """
        raise NotImplementedError
