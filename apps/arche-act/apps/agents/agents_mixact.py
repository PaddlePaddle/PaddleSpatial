# !/usr/bin/env python3
# -*- encoding=utf8 -*-

"""
Gradio-based web app Agents that uses OpenAI/ErnieBot API to generate
a chat between collaborative agents.
"""
import sys
import os
import re
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import openai
import openai.error
import tenacity
from loguru import logger

from text_utils import split_markdown_code
from arche.agents import SpecifyAgent
from arche.messages import BaseMessage
from arche.societies import ArcheActPlaying
from arche.typing import TaskType


logger_config = {
    "handlers": [
        {"sink": sys.stdout, 
         "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
        },
        {"sink": "debug_agents_mixact.log", 
         "backtrace": True, "diagnose": True, "serialize": True, "rotation": "10 MB"},
    ],
    # "extra": {"user": "someone"}
}
logger.configure(**logger_config)
logger.disable("RouteAgent")

REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

ChatBotHistory = List[Tuple[Optional[str], Optional[str]]]
TaskDescpritionHistory = List[Optional[str]]


@dataclass
class State:
    """
    State of the chatbot.
    """
    session: Optional[ArcheActPlaying]
    max_messages: int
    chat: ChatBotHistory
    saved_assistant_msg: Optional[BaseMessage]
    tasks: TaskDescpritionHistory

    @classmethod
    def empty(cls) -> 'State':
        """
        Create an empty state.
        """
        return cls(None, 0, [], None, [])

    @staticmethod
    def construct_inplace(state: 'State', session: Optional[ArcheActPlaying],
                          max_messages: int, chat: ChatBotHistory, 
                          saved_assistant_msg: Optional[BaseMessage], 
                          task_desc: str = None) -> None:
        """
        Construct a state inplace.
        """
        state.session = session
        state.max_messages = max_messages
        state.chat = chat
        state.saved_assistant_msg = saved_assistant_msg
        if task_desc:
            state.tasks.append(task_desc)
        elif task_desc is not None and task_desc == '':
            state.tasks = []
    
    @staticmethod
    def update_tasks(state: 'State', task_desc: str) -> None:
        """
        Update the task descriptions.
        """
        if task_desc:
            state.tasks.append(task_desc)


def parse_arguments():
    """ Get command line arguments. """

    parser = argparse.ArgumentParser("Camel data explorer")
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--eb-access-token', type=str, default=None,
                        help='ErnieBot access token')
    parser.add_argument('--share', type=bool, default=False,
                        help='Expose the web UI to Gradio')
    parser.add_argument('--server-port', type=int, default=8035,
                        help='Port ot run the web page on')
    parser.add_argument('--inbrowser', type=bool, default=False,
                        help='Open the web UI in the default browser on lunch')
    parser.add_argument(
        '--concurrency-count', type=int, default=1,
        help='Number if concurrent threads at Gradio websocket queue. ' +
        'Increase to serve more requests but keep an eye on RAM usage.')
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        logger.error("Unknown args: {}", unknown)
    return args


def load_roles(path: str) -> List[str]:
    """ Load roles from list files.

    Args:
        path (str): Path to the TXT file.

    Returns:
        List[str]: List of roles.
    """

    assert os.path.exists(path)
    roles = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r"^\d+\.\s*(.+)\n*$", line)
            if match:
                role = match.group(1)
                if len(role.split(' # ')) > 1:
                    role = tuple(role.split(' # '))
                roles.append(role)
            else:
                logger.warning("No match role found in {} from {}", line, path)
    return roles


def load_assistant_agent_descs(assistant_agents_path: str) -> Dict[str, str]:
    """ Load executor assistant agent descriptions from list files.

    args:
        assistant_agents_path (str): Path to the TXT file.

    Returns:
        Dict[str, str]: A dict of names and corresponding descriptions.
    """
    assert os.path.exists(assistant_agents_path)
    # names = []
    descs = dict()
    with open(assistant_agents_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r"^\d+\.\s*(.+)\n*$", line)
            if match:
                description = match.group(1)
                if len(description.split(' # ')) > 1:
                    details = tuple(description.split(' # '))
                    agent_name = details[0]
                    # names.append(agent_name)
                    descs.update({agent_name: details[1:]})
            else:
                logger.warning("No match role found in {} from {}", line, assistant_agents_path)
    return descs


def cleanup_on_launch(state) -> Tuple[State, ChatBotHistory, Dict, State]:
    """ Prepare the UI for a new session.

    Args:
        state (State): Role playing state.

    Returns:
        Tuple[State, ChatBotHistory, Dict]:
            - Updated state.
            - Chatbot window contents.
            - Start button state (disabled).
            - Title markdown content (invisible).
    """
    # The line below breaks the every=N runner
    # `state = State.empty()`

    State.construct_inplace(state, None, 0, [], None, '')

    return (state, [], gr.update(interactive=False), 
            gr.update(visible=False))


@logger.catch
def playing_start(
    state,
    society_name: str,
    assistant: str,
    user: str,
    model_names: List[str],
    original_task: str,
    max_messages: float,
    with_task_specifier: bool,
    with_router: bool,
    word_limit: int,
    language: str,
    clarified_task: str = None
) -> Union[Dict, Tuple[State, str, Union[str, Dict], ChatBotHistory, Dict]]:
    """ Creates a Multi-User Multi-Assistant role playing session.

    Args:
        state (State): Session state.
        society_name:
        assistant (str): Contents of the Assistant field.
        user (str): Contents of the User field.
        model_names (List[str]): Contents of the Executive Assistant Agents field.
        original_task (str): Original task field.
        max_messages (float): Max number of messages.
        with_task_specifier (bool): Enable/Disable task specifier.
        with_router (bool): Enable/Disable router agent.
        word_limit (int): Limit of words for task specifier.
        language (str): Output language of the task.
        clarified_task (str): Clarified task field.

    Returns:
        Union[Dict, Tuple[State, str, Union[str, Dict], ChatBotHistory, Dict]]:
            - Updated state.
            - Generated specified task.
            - Planned task (if any).
            - Chatbot window contents.
            - Progress bar contents.
            - Clarified task area state (enable if necessary).
            - Clarify task button state (enable if necessary).
    """

    if state.session is not None:
        logger.info("Double click")
        return {}  # may fail

    if society_name not in {"Travel Assistant", "Trip Plan", "Travel Assistant En"}:
        logger.info("Error: unrecognezed society {}", society_name)
        return {}

    meta_dict: Optional[Dict[str, str]]
    extend_sys_msg_meta_dicts: Optional[List[Dict]]
    task_type: TaskType
    if society_name == "Travel Assistant":
        meta_dict = None
        extend_sys_msg_meta_dicts = None
        # Keep user and assistant intact
        task_type = TaskType.TRAVEL_ASSISTANT
        assistant_agents_subpath = "travel_assistant/assistant_agents.txt"
    elif society_name == "Travel Assistant En":
        meta_dict = None
        extend_sys_msg_meta_dicts = None
        # Keep user and assistant intact
        task_type = TaskType.TRAVEL_ASSISTANT_EN
        assistant_agents_subpath = "travel_assistant_en/assistant_agents.txt"
    else:  # "Trip Plan"
        meta_dict = None
        extend_sys_msg_meta_dicts = None
        # Keep user and assistant intact
        task_type = TaskType.TRIP_PLAN

    try:
        if not model_names:
            raise RuntimeError("Error: No Executive models were selected!")
        
        task_specify_kwargs = dict(word_limit=word_limit) if with_task_specifier else None
        assistant_model_list = []
        assistant_agents_path = os.path.join(REPO_ROOT, f"data/{assistant_agents_subpath}")
        assistant_agent_descs = load_assistant_agent_descs(assistant_agents_path)
        for model_name in model_names:
            model_capability = assistant_agent_descs[model_name]
            assistant_model_list.append([model_name, model_capability])

        if clarified_task:
            State.update_tasks(state, clarified_task)
            if state.tasks:
                original_task = original_task + '\n\n' + '\n\n'.join(desc for desc in state.tasks)
        
        session = ArcheActPlaying(
            assistant,
            user,
            task_prompt=original_task,
            with_task_specify=with_task_specifier,
            task_specify_agent_kwargs=task_specify_kwargs,
            with_task_planner=False,
            assistant_model_list=assistant_model_list,
            with_router=with_router,
            task_type=task_type,
            extend_sys_msg_meta_dicts=extend_sys_msg_meta_dicts,
            extend_task_specify_meta_dict=meta_dict,
            output_language=language,
        )
    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        logger.debug("Agent exception 0 {}", str(ex))
        # traceback.print_exc()
        return (state, str(ex), "", [], gr.update(), gr.update(), gr.update())

    # Can't re-create a state like below since it
    # breaks 'role_playing_chat_cont' runner with every=N.
    # `state = State(session=session, max_messages=int(max_messages), chat=[],`
    # `             saved_assistant_msg=None)`

    State.construct_inplace(state, session, int(max_messages), [], None)
    if session.task_prompt:
        State.update_tasks(state, session.task_prompt.split('问题：')[-1].strip('.。'))

    specified_task_prompt = session.specified_task_prompt \
        if session.specified_task_prompt is not None else ""
    
    if not session.task_is_clarified:
        state.session = None
        return (state, specified_task_prompt, "", [], 
                gr.update(),
                gr.update(interactive=True, visible=True), 
                gr.update(interactive=True, visible=True))
    
    planned_task_prompt = session.planned_task_prompt \
        if session.planned_task_prompt is not None else ""

    planned_task_upd = gr.update(
        value=planned_task_prompt, visible=session.planned_task_prompt is not None)

    progress_update = gr.update(maximum=state.max_messages, value=1,
                                visible=True)

    return (state, specified_task_prompt, planned_task_upd, state.chat,
            progress_update, gr.update(interactive=False), gr.update(interactive=False))


@logger.catch
def playing_chat_init(state) -> \
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
    """ Initialize role playing.

    Args:
        state (State): Role playing state.

    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Chatbot window contents.
            - Progress bar contents.
    """

    if state.session is None:
        logger.debug("Error: session is none on role_playing_chat_init call")
        return state, state.chat, gr.update()

    if not state.session.task_is_clarified:
        return state, [], gr.update()

    session: ArcheActPlaying = state.session

    try:
        init_assistant_msg: BaseMessage
        init_assistant_msg, _ = session.init_chat()
    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        logger.debug("Agent exception 1 {}", str(ex))
        # traceback.print_exc()
        state.session = None
        return state, state.chat, gr.update()

    state.saved_assistant_msg = init_assistant_msg

    progress_update = gr.update(maximum=state.max_messages, value=1,
                                visible=True)

    return state, state.chat, progress_update


# WORKAROUND: do not add type hints for session and chatbot_history
@logger.catch
def playing_chat_cont(state) -> \
        Tuple[State, ChatBotHistory, Dict, Dict]:
    """ Produce a pair of messages by a user and an assistant.
        To be run multiple times.

    Args:
        state (State): Role playing state.

    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Chatbot window contents.
            - Progress bar contents.
            - Start button state (to be eventually enabled).
    """

    if state.session is None:
        return state, state.chat, gr.update(visible=False), gr.update()

    session: ArcheActPlaying = state.session

    if state.saved_assistant_msg is None:
        return state, state.chat, gr.update(), gr.update()

    try:
        assistant_response, user_response = session.step(state.saved_assistant_msg)
        if session.task_is_done(user_response.msg.content, strict_constraint=True):
            state.chat.append((None, split_markdown_code(
                user_response.msg.content.replace('<', '\<').replace('>', '\>')))
            )
            state.session = None
            return state, state.chat, gr.update(visible=False), gr.update(interactive=True)
        if user_response.terminated:
            terminated_reason = '\n'.join(user_response.info["termination_reasons"])
            raise RuntimeError(f"Session is stopped unexpectedly!\n\nPossible reasons are: \n{terminated_reason}\n")
    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        logger.debug("Agent exception 2 {}", str(ex))
        # traceback.print_exc()
        state.session = None
        return state, state.chat, gr.update(visible=False), gr.update(interactive=True)

    if len(user_response.msgs) != 1 or len(assistant_response.msgs) != 1:
        return state, state.chat, gr.update(), gr.update()

    u_msg = user_response.msg
    a_msg = assistant_response.msg

    state.saved_assistant_msg = a_msg

    # logger.debug('u_msg: \n{}\n', u_msg.content)
    # logger.debug('\na_msg: \n{}\n', a_msg.content)

    state.chat.append((None, split_markdown_code(u_msg.content.replace('<', '\<').replace('>', '\>'))))
    state.chat.append((split_markdown_code(a_msg.content), None))

    if len(state.chat) >= state.max_messages:
        state.session = None

    if session.task_is_done(a_msg.content) and session.task_is_done(u_msg.content):
        state.chat.append((None, split_markdown_code("\<TASK_DONE\>")))
        state.session = None

    progress_update = gr.update(maximum=state.max_messages,
                                value=len(state.chat), visible=state.session
                                is not None)

    start_bn_update = gr.update(interactive=state.session is None)

    return (state, state.chat, progress_update, start_bn_update)


def stop_session(state) -> Tuple[State, Dict, Dict]:
    """ Finish the session and leave chat contents as an artefact.
    Args:
        state (State): Role playing state.
    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Progress bar contents.
            - Start button state (to be eventually enabled).
            - Title markdown content (visible).
            - Clarified task area state (disabled).
            - Clarify task button state (disabled).
    """
    state.session = None
    return (state, gr.update(visible=False), gr.update(interactive=True), 
            gr.update(visible=True), 
            gr.update(value='Clarified Task if needed', visible=False),
            gr.update(visible=False))


def construct_ui(blocks, api_key: Optional[str] = None, eb_access_token: Optional[str] = None) -> None:
    """ Build Gradio UI and populate with topics.

    Args:
        api_key (str): OpenAI API key.
        eb_access_token (str): Ernie Bot access token.

    Returns:
        None
    """

    if api_key is not None:
        openai.api_key = api_key

    if eb_access_token is not None:
        os.environ["ERNIE_BOT_ACCESS_TOKEN"] = eb_access_token

    society_dict: Dict[str, Dict[str, Any]] = {}
    for society_name in ("Travel Assistant", "Trip Plan", "Travel Assistant En"):
        if society_name == "Travel Assistant":
            assistant_role_subpath = "travel_assistant/assistant_roles.txt"
            user_role_subpath = "travel_assistant/user_roles.txt"
            assistant_agents_subpath = "travel_assistant/assistant_agents.txt"
            assistant_role = "出行助手"
            user_role = "游客"
            # default_task = "找一条最近的公交路线"
            default_task = "找一条最近的去颐和园的公交路线"
        else:
            assistant_role_subpath = "trip_plan/assistant_roles.txt"
            user_role_subpath = "trip_plan/user_roles.txt"
            assistant_agents_subpath = "trip_plan/assistant_agents.txt"
            assistant_role = "行程规划助手"
            user_role = "游客"
            default_task = "去哈尔滨旅游"

        assistant_role_path = os.path.join(REPO_ROOT, f"data/{assistant_role_subpath}")
        user_role_path = os.path.join(REPO_ROOT, f"data/{user_role_subpath}")
        assistant_agents_path = os.path.join(REPO_ROOT, f"data/{assistant_agents_subpath}")

        society_info = dict(
            assistant_roles=load_roles(assistant_role_path),
            user_roles=load_roles(user_role_path),
            assistant_agents=load_assistant_agent_descs(assistant_agents_path),
            assistant_role=assistant_role,
            user_role=user_role,
            default_task=default_task,
        )
        society_dict[society_name] = society_info

    default_society = society_dict["Travel Assistant"]

    def change_society(society_name: str) -> Tuple[Dict, Dict, str]:
        society = society_dict[society_name]
        assistant_dd_update = gr.update(choices=society['assistant_roles'],
                                        value=society['assistant_role'])
        user_dd_update = gr.update(choices=society['user_roles'],
                                   value=society['user_role'])
        assistant_list_dd_update = gr.update(
                                    choices=list(society["assistant_agents"].keys())
                                    )
        return assistant_dd_update, user_dd_update, society['default_task'], assistant_list_dd_update

    with gr.Row():
        with gr.Column(scale=1):
            society_dd = gr.Dropdown(["Travel Assistant", "Trip Plan"],
                                     label="Choose the society",
                                     value="Travel Assistant", interactive=True)
        with gr.Column(scale=2):
            user_dd = gr.Dropdown(default_society['user_roles'],
                                  label="Example user roles",
                                  value=default_society['user_role'],
                                  interactive=True)
            user_ta = gr.TextArea(label="User role (EDIT ME)", lines=1,
                                  interactive=True)
        with gr.Column(scale=2):
            assistant_dd = gr.Dropdown(default_society['assistant_roles'],
                                       label="Example assistant roles",
                                       value=default_society['assistant_role'],
                                       interactive=True)
            assistant_ta = gr.TextArea(label="Assistant role (EDIT ME)",
                                       lines=1, interactive=True)
        with gr.Column(scale=2):
            assistant_list_dd = gr.Dropdown(list(default_society["assistant_agents"].keys()), 
                                            label="Executive Assistant Agents", 
                                            info="Select candidate executive assistant agents",
                                            multiselect=True,
                                            interactive=True)
    with gr.Row():
        logo_path = "https://github.com/PaddlePaddle/PaddleSpatial/main/apps/arche-act/misc/br_logo.png"
        with gr.Column(scale=2):
            title_md = gr.Markdown(
                "# ArcheAct: A Collaborative Agent Framework with"
                " Disambiguation and Polymorphic Role-Playing\n"
                "Github repo: [https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/arche-act]"
                "(https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/arche-act)"
                '<div style="display:flex; justify-content:center;">'
                f'<img src="{logo_path}"'
                ' alt="Logo" style="max-width:15%;">'
                '</div>',
                visible=True)
    with gr.Row():
        with gr.Column(scale=4):
            original_task_ta = gr.TextArea(
                label="Give me a preliminary idea (EDIT ME)",
                value=default_society['default_task'], lines=1,
                interactive=True)
        with gr.Column(scale=1):
            universal_task_bn = gr.Button("Insert universal task")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                task_specifier_cb = gr.Checkbox(value=True,
                                                label="With task specifier")
            with gr.Row():
                router_cb = gr.Checkbox(value=False,
                                        label="With agent router")
            with gr.Row():
                ts_word_limit_nb = gr.Number(
                    value=SpecifyAgent.DEFAULT_WORD_LIMIT,
                    label="Word limit for task specifier",
                    visible=task_specifier_cb.value)
        with gr.Column():
            with gr.Row():
                num_messages_sl = gr.Slider(minimum=1, maximum=50, step=1,
                                            value=10, 
                                            interactive=True,
                                            label="Messages to generate")
            with gr.Row():
                language_ta = gr.TextArea(label="Language", value="Chinese",
                                          lines=1, 
                                          interactive=True)
        with gr.Column(scale=2):
            with gr.Row():
                start_bn = gr.Button("Make agents chat [takes time]",
                                     elem_id="start_button")
            with gr.Row():
                clear_bn = gr.Button("Interrupt the current query")
    with gr.Row():
        with gr.Column(scale=4):
            clarified_task_ta = gr.TextArea(
                label="Clarify the preliminary idea if necessary (EDIT ME)",
                value='Clarified Task if needed', lines=1,
                interactive=False, visible=False)
        with gr.Column(scale=1):
            clarify_task_bn = gr.Button("Clarify the task", 
                                        interactive=False, visible=False)

    progress_sl = gr.Slider(minimum=0, maximum=100, value=0, step=1,
                            label="Progress", 
                            interactive=False, 
                            visible=False)
    specified_task_ta = gr.TextArea(
        label="Specified task prompt given to the role-playing session"
        " based on the original (simplistic) idea", lines=1, 
        interactive=False)
    task_plan_prompt_ta = gr.TextArea(label="Planned task prompt", lines=1,
                                      interactive=False, 
                                      visible=False)
    
    # ===========  Blocks for role-playing session ===============
    chatbot = gr.Chatbot(label="Chat between autonomous agents")
    empty_state = State.empty()
    session_state: gr.State = gr.State(empty_state)

    universal_task_bn.click(lambda: "Help me to do my job", None,
                            original_task_ta)
    task_specifier_cb.change(lambda v: gr.update(visible=v), task_specifier_cb,
                             ts_word_limit_nb)

    start_bn.click(cleanup_on_launch, session_state,
                   [session_state, chatbot, start_bn, title_md], queue=False) \
            .then(playing_start,
                  [session_state, society_dd, assistant_ta, user_ta, 
                   assistant_list_dd, original_task_ta, num_messages_sl,
                   task_specifier_cb, router_cb, ts_word_limit_nb, language_ta],
                  [session_state, specified_task_ta, task_plan_prompt_ta,
                   chatbot, progress_sl, clarified_task_ta, clarify_task_bn],
                  queue=False) \
            .then(playing_chat_init, session_state,
                  [session_state, chatbot, progress_sl], 
                  queue=False)
    
    clarify_task_bn.click(playing_start,
                  [session_state, society_dd, assistant_ta, user_ta, 
                   assistant_list_dd, original_task_ta, num_messages_sl,
                   task_specifier_cb, router_cb, ts_word_limit_nb, language_ta, 
                   clarified_task_ta],
                  [session_state, specified_task_ta, task_plan_prompt_ta,
                   chatbot, progress_sl, clarified_task_ta, clarify_task_bn],
                  queue=False) \
            .then(playing_chat_init, session_state,
                  [session_state, chatbot, progress_sl], 
                  queue=False)

    blocks.load(playing_chat_cont, session_state,
                [session_state, chatbot, progress_sl, start_bn], every=0.5)

    clear_bn.click(stop_session, session_state,
                   [session_state, progress_sl, start_bn, title_md, 
                    clarified_task_ta, clarify_task_bn])

    society_dd.change(change_society, society_dd,
                      [assistant_dd, user_dd, original_task_ta, assistant_list_dd])
    assistant_dd.change(lambda dd: dd, assistant_dd, assistant_ta)
    user_dd.change(lambda dd: dd, user_dd, user_ta)

    blocks.load(change_society, society_dd,
                [assistant_dd, user_dd, original_task_ta, assistant_list_dd])
    blocks.load(lambda dd: dd, assistant_dd, assistant_ta)
    blocks.load(lambda dd: dd, user_dd, user_ta)


def construct_blocks(api_key: Optional[str]=None, eb_access_token: Optional[str]=None):
    """ Construct Agents app but do not launch it.

    Args:
        api_key (Optional[str]): OpenAI API key.
        eb_access_token (Optional[str]): Ernie Bot access token.

    Returns:
        gr.Blocks: Blocks instance.
    """

    css_str = "#start_button {border: 3px solid #4CAF50; font-size: 20px;}"

    with gr.Blocks(css=css_str) as blocks:
        if eb_access_token is None:
            construct_ui(blocks, api_key=api_key)
        else:
            construct_ui(blocks, eb_access_token=eb_access_token)

    return blocks


def main():
    """ Entry point. """

    args = parse_arguments()

    logger.info("Getting Agents web server online...")

    if args.eb_access_token is None:
        blocks = construct_blocks(api_key=args.api_key)
    else:
        blocks = construct_blocks(eb_access_token=args.eb_access_token)

    blocks.queue(args.concurrency_count) \
          .launch(share=args.share, inbrowser=args.inbrowser,
                  server_name="0.0.0.0", server_port=args.server_port,
                  debug=True)

    logger.info("Exiting.")


if __name__ == "__main__":
    main()
