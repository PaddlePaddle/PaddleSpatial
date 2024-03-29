# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Dict, List, Optional

from colorama import Fore

from arche.agents import (
    BaseToolAgent,
    ChatAgent,
    ChatAgentResponse,
    HuggingFaceToolAgent,
)
from arche.messages import BaseMessage
from arche.typing import ModelType
from arche.utils import PythonInterpreter, print_text_animated


class EmbodiedAgent(ChatAgent):
    r"""Class for managing conversations of CAMEL Embodied Agents.

    Args:
        system_message (BaseMessage): The system message for the chat agent.
        model (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_4`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
        action_space (List[Any], optional): The action space for the embodied
            agent. (default: :obj:`None`)
        verbose (bool, optional): Whether to print the critic's messages.
        logger_color (Any): The color of the logger displayed to the user.
            (default: :obj:`Fore.MAGENTA`)
    """

    def __init__(
        self,
        system_message: BaseMessage,
        model: ModelType = ModelType.GPT_4,
        model_config: Optional[Any] = None,
        message_window_size: Optional[int] = None,
        action_space: Optional[List[BaseToolAgent]] = None,
        verbose: bool = False,
        logger_color: Any = Fore.MAGENTA,
    ) -> None:
        default_action_space = [
            HuggingFaceToolAgent('hugging_face_tool_agent', model=model.value),
        ]
        self.action_space = action_space or default_action_space
        action_space_prompt = self.get_action_space_prompt()
        system_message.content = system_message.content.format(
            action_space=action_space_prompt)
        self.verbose = verbose
        self.logger_color = logger_color
        super().__init__(
            system_message=system_message,
            model=model,
            model_config=model_config,
            message_window_size=message_window_size,
        )

    def get_action_space_prompt(self) -> str:
        r"""Returns the action space prompt.

        Returns:
            str: The action space prompt.
        """
        return "\n".join([
            f"*** {action.name} ***:\n {action.description}"
            for action in self.action_space
        ])

    def step(
        self,
        input_message: BaseMessage,
    ) -> ChatAgentResponse:
        r"""Performs a step in the conversation.

        Args:
            input_message (BaseMessage): The input message.

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        response = super().step(input_message)

        if response.msgs is None or len(response.msgs) == 0:
            raise RuntimeError("Got None output messages.")
        if response.terminated:
            raise RuntimeError(f"{self.__class__.__name__} step failed.")

        # NOTE: Only single output messages are supported
        explanations, codes = response.msg.extract_text_and_code_prompts()

        if self.verbose:
            for explanation, code in zip(explanations, codes):
                print_text_animated(self.logger_color +
                                    f"> Explanation:\n{explanation}")
                print_text_animated(self.logger_color + f"> Code:\n{code}")

            if len(explanations) > len(codes):
                print_text_animated(self.logger_color +
                                    f"> Explanation:\n{explanations}")

        content = response.msg.content

        if codes is not None:
            content = "\n> Executed Results:"
            action_space: Dict[str, Any] = {
                action.name: action
                for action in self.action_space
            }
            action_space.update({"print": print, "enumerate": enumerate})
            interpreter = PythonInterpreter(action_space=action_space)
            for block_idx, code in enumerate(codes):
                executed_outputs, _ = code.execute(interpreter)
                content += (f"Executing code block {block_idx}:\n"
                            f"  - execution output:\n{executed_outputs}\n"
                            f"  - Local variables:\n{interpreter.state}\n")
                content += "*" * 50 + "\n"

        # TODO: Handle errors
        content = input_message.content + (Fore.RESET +
                                           f"\n> Embodied Actions:\n{content}")
        message = BaseMessage(input_message.role_name, input_message.role_type,
                              input_message.meta_dict, content)
        return ChatAgentResponse([message], response.terminated, response.info)
