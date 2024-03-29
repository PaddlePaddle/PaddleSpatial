# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

import gradio as gr
import pytest

from apps.agents.agents_mixact import (
    State,
    cleanup_on_launch,
    construct_blocks,
    parse_arguments,
    playing_chat_cont,
    playing_chat_init,
    playing_start,
    stop_session,
)


def test_construct_blocks():
    """
    Test the construction of blocks.
    """
    blocks = construct_blocks(None)
    assert isinstance(blocks, gr.Blocks)


def test_utils():
    """
        Test the utils.
    """
    args = parse_arguments()
    assert args is not None


@pytest.mark.model_backend
def test_session():
    """
    Desc:
    `   test session
    """
    for society_name in ("Travel Assistant"):

        state = State.empty()

        state, _, _ = cleanup_on_launch(state)

        if society_name == "Travel Assistant":
            assistant = "Travel Robot"
            user = "Traveler"
            original_task = "Find a nearest restaurant"
        else:
            assistant = "JavaScript"
            user = "Sociology"
            original_task = "Develop a poll app"

        max_messages = 10
        with_task_specifier = False
        assistant_agent_list = ["EB_TURBO", "EB_8K"]
        with_router = False 
        word_limit = 50
        language = "English"
        state, specified_task_prompt, planned_task_upd, chat, progress_upd = \
            playing_start(state, society_name, assistant, user, assistant_agent_list,
                          original_task, max_messages,
                          with_task_specifier, with_router, word_limit, language)

        assert state.session is not None

        state, chat, progress_update = \
            playing_chat_init(state)

        assert state.session is not None

        for _ in range(5):
            state, chat, progress_update, start_bn_update =\
                playing_chat_cont(state)

        state, _, _ = stop_session(state)

        assert state.session is None
