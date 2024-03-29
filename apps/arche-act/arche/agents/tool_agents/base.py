# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from arche.agents import BaseAgent


class BaseToolAgent(BaseAgent):
    r"""Creates a :obj:`BaseToolAgent` object with the specified name and
        description.

    Args:
        name (str): The name of the tool agent.
        description (str): The description of the tool agent.
    """

    def __init__(self, name: str, description: str) -> None:

        self.name = name
        self.description = description

    def reset(self) -> None:
        r"""Resets the agent to its initial state."""
        pass

    def step(self) -> None:
        r"""Performs a single step of the agent."""
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
