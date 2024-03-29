# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential
from loguru import logger

from arche.configs import BaseConfig
from arche.functions import ErnieBotFunction
from arche.messages import BaseMessage, FunctionCallingMessage, OpenAIMessage, ErnieBotMessage
from arche.models import BaseModelBackend, ModelFactory
from arche.typing import ModelType, RoleType
from arche.utils import openai_api_key_required, cast_to_model_type
from arche.agents import ChatRecord, ChatAgentResponse, ChatAgent, FunctionCallingRecord, get_model_config


class AssistantAgent(ChatAgent):
    r"""The class of assistant agent.

    Args:
        name (str): The name of an agent instance.
    """
    def __init__(self, 
        system_message: BaseMessage,
        assistant_model_list: Optional[List],
        message_window_size: Optional[int] = 0,
        output_language: Optional[str] = None,
        function_list: Optional[List[ErnieBotFunction]] = None,
        with_declare_output_language: bool = False,
    ):
        # e.g. {'model_name': {'model_type': ModelType, 'model_config': ModelConfig, 'model_capability': Description}}
        self.models: Dict(str, Any) = dict()
        for model_name, model_capability in assistant_model_list:
            model_type = cast_to_model_type(model_name)
            model_name = model_name.strip().lower()
            self.models.update(
                {model_name: 
                    dict(
                        model_type=model_type,
                        model_capability=model_capability,
                        model_config=get_model_config(model_type),
                    )
                }
            )
        self.model: Optional[ModelType] = None
        self.model_config: Optional[BaseConfig] = None
        self.model_backend: BaseModelBackend = None
        self.model_token_limit: int = 0
        self.init_model()

        super().__init__(system_message=system_message, 
                         model=self.model, model_config=self.model_config, 
                         message_window_size=message_window_size, output_language=output_language, 
                         function_list=function_list, with_declare_output_language=with_declare_output_language)
    
    def init_model(self, 
                   model: Optional[ModelType] = None, 
                   is_reset: bool = False
    ):
        r"""Instantiates a model backend if necessary.
        """
        assert len(self.models.keys()) > 0, "No model is provided."
        self.model: ModelType = \
            model if model is not None else list(self.models.items())[0][1]["model_type"]

        self.model_config = self.models[self.model.name.lower()]["model_config"]
        
        logger.debug(f'\nAssistantAgent.model_config: {self.model_config.__dict__}\n')

        if is_reset:
            self.model_backend: BaseModelBackend = ModelFactory.create(
                self.model, self.model_config.__dict__)
            self.model_token_limit: int = self.model_backend.token_limit
    
    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(1))
    @openai_api_key_required
    def step(
        self,
        input_message: BaseMessage, 
        exec_model: Optional[ModelType] = None, 
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message.

        Args:
            input_message (BaseMessage): The input message to the agent.
            Its `role` field that specifies the role at backend may be either
            `user` or `assistant` but it will be set to `user` anyway since
            for the self agent any incoming message is external.
            exec_model (ModelType): The model to be used for generating the response.

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """

        logger.debug('>>> AssistantAgent.step() >>>')
        messages = self.declare_output_language_in_msg(input_message)
        
        # print(f'After updating messages: \n{messages}\n')

        self.init_model(model=exec_model, is_reset=True)
        output_messages: List[BaseMessage]
        info: Dict[str, Any]
        called_funcs: List[FunctionCallingRecord] = []
        while True:
            # Format messages and get the token number
            prep_messages: Optional[List[ErnieBotMessage]]
            num_tokens: int

            prep_messages, num_tokens = self.preprocess_messages(messages)

            logger.debug(f'Preprocessed messages: {prep_messages}')
            logger.debug(f'number of tokens {num_tokens}')

            # Terminate when number of tokens exceeds the limit
            if num_tokens >= self.model_token_limit:
                return self.step_token_exceed(num_tokens, called_funcs)

            # Obtain LLM's response and validate it
            response = self.model_backend.run(prep_messages)
            self.validate_model_response(response)

            if not self.model_backend.stream:
                output_messages, finish_reasons, usage_dict, response_id = (
                    self.handle_batch_response(response))
            else:
                output_messages, finish_reasons, usage_dict, response_id = (
                    self.handle_stream_response(response, num_tokens))

            if self.is_function_calling_enabled() and finish_reasons[0] == 'function_call':
                # Do function calling
                func_assistant_msg, func_result_msg, func_record = (
                    self.step_function_call(response))

                # Update the messages
                messages = self.update_messages('assistant', func_assistant_msg)
                messages = self.update_messages('function', func_result_msg)
                called_funcs.append(func_record)
            else:
                # Function calling disabled or chat stopped
                info = self.get_info(
                    response_id,
                    usage_dict,
                    finish_reasons,
                    num_tokens,
                    called_funcs,
                )
                break

        return ChatAgentResponse(output_messages, self.terminated, info)
    
    def preprocess_messages(
            self,
            messages: List[ChatRecord]) -> Tuple[List[OpenAIMessage], int]:
        r"""Convert the list of messages to adapt to different model's format and calculate
        the number of tokens accordingly.

        Args:
            messages (List[ChatRecord]): The list of structs containing
                information about previous chat messages.

        Returns:
            tuple: A tuple containing the truncated list of messages in
                OpenAI's input format and the number of tokens.
        """
        if self.model.is_openai:
            return self.preprocess_openai_messages(messages)
        elif self.model.is_ernie_bot:
            return self.preprocess_eb_messages(messages)
        else:
            raise NotImplementedError(f"Cannot preprocess the messages for model {self.model}!")

    def handle_batch_response(
        self, response: Dict[str, Any]
    ) -> Tuple[List[BaseMessage], List[str], Dict[str, int], str]:
        r"""Handle the batch response adpatively from a model.
        Args:
            response (dict): Model response.
        Returns:
            tuple: A tuple of list of output `ChatMessage`, list of
                finish reasons, usage dictionary, and response id.
        """
        if self.model.is_openai:
            return self.handle_openai_batch_response(response)
        elif self.model.is_ernie_bot:
            return self.handle_eb_batch_response(response)
        else:
            raise NotImplementedError(f"Cannot handle the response generated by the model {self.model}!")

    def handle_stream_response(
        self,
        response: Any,
        prompt_tokens: int,
    ) -> Tuple[List[BaseMessage], List[str], Dict[str, int], str]:
        r"""Handle the stream response generated by a model accordingly.

        Args:
            response (dict): Model response.
            prompt_tokens (int): Number of input prompt tokens.

        Returns:
            tuple: A tuple of list of output `ChatMessage`, list of
                finish reasons, usage dictionary, and response id.
        """
        if self.model.is_openai:
            return self.handle_openai_stream_response(response, prompt_tokens)
        else:
            raise NotImplementedError(f"Cannot handle the response generated by the model {self.model}!")
