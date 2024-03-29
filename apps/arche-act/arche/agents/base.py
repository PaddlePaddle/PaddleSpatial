# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    r"""An abstract base class for all CAMEL agents."""

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        r"""Resets the agent to its initial state."""
        pass

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        r"""Performs a single step of the agent."""
        pass
