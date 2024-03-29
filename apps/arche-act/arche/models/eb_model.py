# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""
import os
from types import GeneratorType
from typing import Any, Dict, List, Optional
from loguru import logger

from arche.configs import EB_API_PARAMS, EB_API_PARAMS_WITH_FUNCTIONS
from arche.messages import ErnieBotMessage
from arche.models import BaseModelBackend
from arche.typing import ModelType
from arche.utils import BaseTokenCounter, ErnieBotTokenCounter, OpenAITokenCounter


DEFAULT_API_BASE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/"


class ErnieBotModel(BaseModelBackend):
    r"""Ernie-Bot API in a unified BaseModelBackend interface."""

    def __init__(self, model_type: ModelType, 
                 model_config_dict: Dict[str, Any]) -> None:
        r"""Constructor for ErnieBot backend.

        Args:
            model_type (ModelType): Model for which a backend is created,
                one of EB_* series.
            model_config_dict (Dict[str, Any]): A dictionary that will
                be fed into the Ernie-Bot API.
        """
        super().__init__(model_type, model_config_dict)
        self._token_counter: Optional[BaseTokenCounter] = None
        self.model_type = model_type

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        eb_access_token: str = os.environ.get("ERNIE_BOT_ACCESS_TOKEN")
        if not self._token_counter:
            self._token_counter = ErnieBotTokenCounter(self.model_type, access_token=eb_access_token)
        return self._token_counter

    def run(
        self,
        messages: List[Dict],
    ) -> Dict[str, Any]:
        r"""Run inference of ErnieBot chat completion.

        Args:
            messages (List[Dict]): Message list with the chat history
                in ErnieBot API format.

        Returns:
            Dict[str, Any]: Response in the ErnieBot API format.
        """
        response = self.chat_completion(messages=messages)
        if not self.stream:
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected batch return from ErnieBot API")
        else:
            if not isinstance(response, GeneratorType):
                raise RuntimeError("Unexpected stream return from ErnieBot API")
        return response
    
    def chat_completion(self, messages: List[Dict]) -> Dict[str, Any]:
        r"""
        """
        logger.debug('\n>>> (model.run()) \nErnieBotModel.chat_completion >>>\n')

        import requests, json
        api_base = DEFAULT_API_BASE
        eb_access_token: str = os.environ.get("ERNIE_BOT_ACCESS_TOKEN")
        api_full = f"{api_base}{self.model_type.api_for_eb_model}?access_token={eb_access_token}"

        logger.debug(f"\nFULL_API: {api_full}\n")
        
        headers = {
            'Content-Type': 'application/json'
        }
        messages_eb: List[ErnieBotMessage] = messages

        # print(f'Original message_eb: {messages_eb}')
        # print(f'\nLength of original message_eb: {len(messages_eb)}\n')

        messages_eb_tailored = []
        for msg in messages_eb:
            if 'role' in msg and msg['role'] == 'system':
                continue
            messages_eb_tailored.append(msg)
        
        logger.debug(f'Tailored message_eb: {messages_eb_tailored}')
        logger.debug(f'\nLength of tailored message_eb: {len(messages_eb_tailored)}\n')

        msg_to_api = {"messages": messages_eb_tailored}
        msg_to_api.update(**self.model_config_dict)

        response = requests.request("POST", api_full, headers=headers, data=json.dumps(msg_to_api))
        if response.status_code == 200:
            dic_res = json.loads(response.text)
            if 'error_code' in dic_res:
                raise NotImplementedError(f"err_code: {dic_res['error_code']}, err_msg: {dic_res['error_msg']}")
            # return dic_res['result']
            logger.debug(f'type of response: {type(response)}')
            logger.debug(f'response: {dic_res}')
            # print(f'response.keys(): {dic_res.keys()}')
            for key in dic_res:
                logger.debug(f'key: {key}')
                logger.debug(f'value: {dic_res[key]}\n')
            return dic_res
        else:
            raise Exception(f'Invalid status code ({response.status_code}) is returned!')

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to ErnieBot API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to ErnieBot API.
        """
        # print(f'EB_API_PARAMS: {EB_API_PARAMS}')
        # print(f'EB_API_PARAMS_WITH_FUNCTIONS: {EB_API_PARAMS_WITH_FUNCTIONS}')

        for param in self.model_config_dict:
            if param not in EB_API_PARAMS_WITH_FUNCTIONS:
                raise ValueError(f"Unexpected argument `{param}` is "
                                 "input into ErnieBot model backend.")

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode,
            which sends partial results each time.
        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
