# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import random
import warnings
from typing import Any, Dict, Optional, Sequence

from colorama import Fore

from arche.agents import ChatAgent, ChatAgentResponse
from arche.messages import BaseMessage
from arche.typing import ModelType
from arche.utils import get_first_int, print_text_animated


class CriticAgent(ChatAgent):
    r"""A class for the critic agent that assists in selecting an option.

    Args:
        system_message (BaseMessage): The system message for the critic
            agent.
        model (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`6`)
        retry_attempts (int, optional): The number of retry attempts if the
            critic fails to return a valid option. (default: :obj:`2`)
        verbose (bool, optional): Whether to print the critic's messages.
        logger_color (Any): The color of the menu options displayed to the
            user. (default: :obj:`Fore.MAGENTA`)
    """

    def __init__(
        self,
        system_message: BaseMessage,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Optional[Any] = None,
        message_window_size: int = 6,
        retry_attempts: int = 2,
        verbose: bool = False,
        logger_color: Any = Fore.MAGENTA,
    ) -> None:
        super().__init__(system_message, model=model,
                         model_config=model_config,
                         message_window_size=message_window_size)
        self.options_dict: Dict[str, str] = dict()
        self.retry_attempts = retry_attempts
        self.verbose = verbose
        self.logger_color = logger_color

    def flatten_options(self, messages: Sequence[BaseMessage]) -> str:
        r"""Flattens the options to the critic.

        Args:
            messages (Sequence[BaseMessage]): A list of `BaseMessage` objects.

        Returns:
            str: A string containing the flattened options to the critic.
        """
        options = [message.content for message in messages]
        flatten_options = (
            f"> Proposals from "
            f"{messages[0].role_name} ({messages[0].role_type}). "
            "Please choose an option:\n")
        for index, option in enumerate(options):
            flatten_options += f"Option {index + 1}:\n{option}\n\n"
            self.options_dict[str(index + 1)] = option
        format = (
            f"Please first enter your choice ([1-{len(self.options_dict)}]) "
            "and then your explanation and comparison: ")
        return flatten_options + format

    def get_option(self, input_message: BaseMessage) -> str:
        r"""Gets the option selected by the critic.

        Args:
            input_message (BaseMessage): A `BaseMessage` object representing
                the input message.

        Returns:
            str: The option selected by the critic.
        """
        # TODO: Add support for editing options by the critic.
        msg_content = input_message.content
        i = 0
        while i < self.retry_attempts:
            critic_response = self.step(input_message)

            if critic_response.msgs is None or len(critic_response.msgs) == 0:
                raise RuntimeError("Got None critic messages.")
            if critic_response.terminated:
                raise RuntimeError("Critic step failed.")

            critic_msg = critic_response.msg
            self.update_messages('assistant', critic_msg)
            if self.verbose:
                print_text_animated(self.logger_color + "\n> Critic response: "
                                    f"\x1b[3m{critic_msg.content}\x1b[0m\n")
            choice = self.parse_critic(critic_msg)

            if choice in self.options_dict:
                return self.options_dict[choice]
            else:
                input_message = BaseMessage(
                    role_name=input_message.role_name,
                    role_type=input_message.role_type,
                    meta_dict=input_message.meta_dict,
                    content="> Invalid choice. Please choose again.\n" +
                    msg_content,
                )
                i += 1
        warnings.warn("Critic failed to get a valid option. "
                      f"After {self.retry_attempts} attempts. "
                      "Returning a random option.")
        return random.choice(list(self.options_dict.values()))

    def parse_critic(self, critic_msg: BaseMessage) -> Optional[str]:
        r"""Parses the critic's message and extracts the choice.

        Args:
            critic_msg (BaseMessage): A `BaseMessage` object representing the
                critic's response.

        Returns:
            Optional[str]: The critic's choice as a string, or None if the
                message could not be parsed.
        """
        choice = str(get_first_int(critic_msg.content))
        return choice

    def reduce_step(
        self,
        input_messages: Sequence[BaseMessage],
    ) -> ChatAgentResponse:
        r"""Performs one step of the conversation by flattening options to the
        critic, getting the option, and parsing the choice.

        Args:
            messages (Sequence[BaseMessage]): A list of BaseMessage objects.

        Returns:
            ChatAgentResponse: A `ChatAgentResponse` object includes the
                critic's choice.
        """
        meta_chat_message = BaseMessage(
            role_name=input_messages[0].role_name,
            role_type=input_messages[0].role_type,
            meta_dict=input_messages[0].meta_dict,
            content="",
        )

        flatten_options = self.flatten_options(input_messages)
        if self.verbose:
            print_text_animated(self.logger_color +
                                f"\x1b[3m{flatten_options}\x1b[0m\n")
        input_msg = meta_chat_message.create_new_instance(flatten_options)

        option = self.get_option(input_msg)
        output_msg = meta_chat_message.create_new_instance(option)

        # TODO: The return `info` can be improved.
        return ChatAgentResponse([output_msg], terminated=False, info={})
