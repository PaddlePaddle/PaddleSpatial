# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import inspect
import os
import re
import socket
import time
import zipfile
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

import requests

from arche.typing import ModelType, TaskType

F = TypeVar('F', bound=Callable[..., Any])


def openai_api_key_required(func: F) -> F:
    r"""Decorator that checks if the OpenAI API key is available in the
    environment variables.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The decorated function.

    Raises:
        ValueError: If the OpenAI API key is not found in the environment
            variables.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from arche.agents.chat_agent import ChatAgent
        if not isinstance(self, ChatAgent):
            raise ValueError("Expected ChatAgent")
        if self.model == ModelType.STUB:
            return func(self, *args, **kwargs)
        elif self.model.is_open_source:
            return func(self, *args, **kwargs)
        elif self.model.is_ernie_bot:
            return func(self, *args, **kwargs)
        elif 'OPENAI_API_KEY' in os.environ:
            return func(self, *args, **kwargs)
        else:
            raise ValueError('OpenAI API key not found.')

    return cast(F, wrapper)


def print_text_animated(text, delay: float = 0.02, end: str = ""):
    r"""Prints the given text with an animated effect.

    Args:
        text (str): The text to print.
        delay (float, optional): The delay between each character printed.
            (default: :obj:`0.02`)
        end (str, optional): The end character to print after each
            character of text. (default: :obj:`""`)
    """
    for char in text:
        print(char, end=end, flush=True)
        time.sleep(delay)
    print('\n')


def get_prompt_template_key_words(template: str) -> Set[str]:
    r"""Given a string template containing curly braces {}, return a set of
    the words inside the braces.

    Args:
        template (str): A string containing curly braces.

    Returns:
        List[str]: A list of the words inside the curly braces.

    Example:
        >>> get_prompt_template_key_words('Hi, {name}! How are you {status}?')
        {'name', 'status'}
    """
    return set(re.findall(r'{([^}]*)}', template))


def get_first_int(string: str) -> Optional[int]:
    r"""Returns the first integer number found in the given string.

    If no integer number is found, returns None.

    Args:
        string (str): The input string.

    Returns:
        int or None: The first integer number found in the string, or None if
            no integer number is found.
    """
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None


def download_tasks(task: TaskType, folder_path: str) -> None:
    """
    Download the tasks from the Google Drive link.
    """
    # Define the path to save the zip file
    zip_file_path = os.path.join(folder_path, "tasks.zip")

    # Download the zip file from the Google Drive link
    response = requests.get("https://huggingface.co/datasets/camel-ai/"
                            f"metadata/resolve/main/{task.value}_tasks.zip")

    # Save the zip file
    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    # Delete the zip file
    os.remove(zip_file_path)


def parse_doc(func: Callable) -> Dict[str, Any]:
    r"""Parse the docstrings of a function to extract the function name,
    description and parameters.

    Args:
        func (Callable): The function to be parsed.
    Returns:
        Dict[str, Any]: A dictionary with the function's name,
            description, and parameters.
    """

    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(
            f"Invalid function {func.__name__}: no docstring provided.")

    properties = {}
    required = []

    parts = re.split(r'\n\s*\n', doc)
    func_desc = parts[0].strip()

    args_section = next((p for p in parts if 'Args:' in p), None)
    if args_section:
        args_descs: List[Tuple[str, str, str, ]] = re.findall(
            r'(\w+)\s*\((\w+)\):\s*(.*)', args_section)
        properties = {
            name.strip(): {
                'type': type,
                'description': desc
            }
            for name, type, desc in args_descs
        }
        for name in properties:
            required.append(name)

    # Parameters from the function signature
    sign_params = list(inspect.signature(func).parameters.keys())
    if len(sign_params) != len(required):
        raise ValueError(
            f"Number of parameters in function signature ({len(sign_params)})"
            f" does not match that in docstring ({len(required)}).")

    for param in sign_params:
        if param not in required:
            raise ValueError(f"Parameter '{param}' in function signature"
                             " is missing in the docstring.")

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Construct the function dictionary
    function_dict = {
        "name": func.__name__,
        "description": func_desc,
        "parameters": parameters,
    }

    return function_dict


def get_task_list(task_response: str) -> List[str]:
    r"""Parse the response of the Agent and return task list.

    Args:
        task_response (str): The string response of the Agent.

    Returns:
        List[str]: A list of the string tasks.
    """

    new_tasks_list = []
    task_string_list = task_response.strip().split('\n')
    # each task starts with #.
    for task_string in task_string_list:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
    return new_tasks_list


def check_server_running(server_url: str) -> bool:
    r"""Check whether the port refered by the URL to the server
    is open.

    Args:
        server_url (str): The URL to the server running LLM inference
            service.

    Returns:
        bool: Whether the port is open for packets (server is running).
    """
    parsed_url = urlparse(server_url)
    url_tuple = (parsed_url.hostname, parsed_url.port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(url_tuple)
    sock.close()

    # if the port is open, the result should be 0.
    return result == 0


def cast_to_model_type(model_name: str) -> ModelType:
    r"""Casts a model name to the corresponding model type.
    Args:
        model_name (str): The model name.
    Returns:
        ModelType: The model type.
    Raises:
        ValueError: If there is no backend for the model.
    """
    if model_name.strip().lower() == ModelType.GPT_3_5_TURBO.name.lower():
        return ModelType.GPT_3_5_TURBO
    elif model_name.strip().lower() == ModelType.GPT_3_5_TURBO_16K.name.lower():
        return ModelType.GPT_3_5_TURBO_16K
    elif model_name.strip().lower() == ModelType.GPT_4.name.lower():
        return ModelType.GPT_4
    elif model_name.strip().lower() == ModelType.GPT_4_32k.name.lower():
        return ModelType.GPT_4_32k
    elif model_name.strip().lower() == ModelType.EB.name.lower():
        return ModelType.EB
    elif model_name.strip().lower() == ModelType.EB_TURBO.name.lower():
        return ModelType.EB_TURBO
    elif model_name.strip().lower() == ModelType.EB_8K.name.lower():
        return ModelType.EB_8K
    elif model_name.strip().lower() == ModelType.EB_4.name.lower():
        return ModelType.EB_4
    elif model_name.strip().lower() == ModelType.LLAMA_2.name.lower():
        return ModelType.LLAMA_2
    elif model_name.strip().lower() == ModelType.VICUNA.name.lower():
        return ModelType.VICUNA
    elif model_name.strip().lower() == ModelType.VICUNA_16K.name.lower():
        return ModelType.VICUNA_16K
    elif model_name.strip().lower() == ModelType.STUB.name.lower():
        return ModelType.STUB
    else:
        raise ValueError(f"Ceasting Error! No backend model is available for {model_name}!")
