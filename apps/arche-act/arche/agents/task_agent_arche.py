# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Dict, List, Optional, Union
from loguru import logger

from arche.agents import ChatAgent
from arche.configs import ChatGPTConfig, ErnieBotConfig
from arche.messages import BaseMessage
from arche.prompts import PromptTemplateGenerator, TextPrompt
from arche.typing import ModelType, RoleType, TaskType
from arche.utils import get_task_list


def format_name_list(name_list: list, output_language: str = None) -> str:
    """Format a list of names into a string.
    Args:
        name_list (List): The name list to be formatted.
        output_language (str, optional): The language to be output by the
            agents. (default: :obj:`None`)
    Returns:
        The formatted name list in a string format.
    """
    if not isinstance(name_list, list):
        raise ValueError("Name list must be a list.")
    sep = '和' \
        if output_language.lower() in 'chinese' or 'chinese' in output_language.lower() \
        else ' and '
    name_list_str = ', '.join(name_list[:-2]) 
    name_list_str += ', ' + sep.join(name_list[-2:]) if name_list_str else sep.join(name_list[-2:])
    return name_list_str


def format_assistant_agent_list(assistant_model_list: List[str], 
                                original_prompt: TextPrompt, 
                                output_language: str) -> TextPrompt:
    r"""Format the assistant agent list into a string.
    Args:
        assistant_model_list (List[str]): A list of models to be used by the
        assistant agent.
        original_prompt (TextPrompt): The original prompt.
        output_language (str): The language to be output by the agent.
    Returns:
        The formatted assistant agent list in a string format.
    """
    upd_prompt = original_prompt
    # logger.info(f"original prompt: {original_prompt}")
    # logger.info(f"assistant_model_list: {assistant_model_list}")
    if assistant_model_list:
        assistant_agent_names = [agent_name for agent_name, _ in assistant_model_list]
        assistant_agents = format_name_list(assistant_agent_names, output_language)
        assistant_agent_details = [': '.join([agent_name, agent_desc[0]]) 
                                for agent_name, agent_desc in assistant_model_list]
        assistant_agent_details = '\n'.join(item for item in assistant_agent_details)
        # logger.info(f"assistant_agents: {assistant_agents}")
        # logger.info(f"assistant_agent_details: {assistant_agent_details}")
        upd_prompt = original_prompt.format(
                                            assistant_agents=assistant_agents, 
                                            assistant_agent_details=assistant_agent_details
                                        )
    # logger.info(f"Updated prompt: {upd_prompt}")
    return upd_prompt


class JudgeAgent(ChatAgent):
    r"""An agent that judges the input statement is True or False according to 
    the given descriptions or knowledge.

    Args:
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.EB_4`)
        task_type (TaskType, optional): The type of task for which to generate
            a prompt. (default: :obj:`TaskType.TRAVEL_ASSISTANT`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        judgement_prompt (Union[str, TextPrompt], optional): The prompt for
            judging the statement. (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agent. (default: :obj:`None`)
    """
    def __init__(
        self,
        model: Optional[ModelType] = None,
        task_type: TaskType = TaskType.TRAVEL_ASSISTANT,
        model_config: Optional[Any] = None,
        judgement_prompt: Optional[Union[str, TextPrompt]] = None,
        output_language: Optional[str] = None,
    ) -> None:
        self.judgement_prompt: Union[str, TextPrompt]
        if judgement_prompt is None:
            judgement_prompt_template = \
                PromptTemplateGenerator().get_prompt_from_key(task_type, "judge_prompt")
            self.judgement_prompt = judgement_prompt_template
        else:
            self.judgement_prompt = TextPrompt(judgement_prompt)

        model_config = model_config or ErnieBotConfig(temperature=0.05)

        if output_language.lower() in 'chinese' or 'chinese' in output_language.lower():
            sys_prompt = "请做出判断。"
        else:
            sys_prompt = "You can judge a statement."
        
        system_message = BaseMessage(
            role_name="Statement Judge",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=sys_prompt        
        )

        super().__init__(system_message, 
                         model=model,
                         model_config=model_config,
                         with_declare_output_language=False)

    def run(self,
        source_prompt: Union[str, TextPrompt],
        statement_prompt: Union[str, TextPrompt],
    ) -> TextPrompt:
        r"""Specify the given task prompt by providing more details.

        Args:
            task_prompt (Union[str, TextPrompt]): The original task
                prompt.
            meta_dict (Dict[str, Any], optional): A dictionary containing
                additional information to include in the prompt.
                (default: :obj:`None`)

        Returns:
            TextPrompt: The specified task prompt.
        """
        self.reset()
        judge_prompt = self.judgement_prompt.format(
                                                    description=source_prompt, 
                                                    statement=statement_prompt
                                                    )

        task_msg = BaseMessage.make_user_message(role_name="Statement Judge",
                                                 content=judge_prompt)
        judge_response = self.step(task_msg)
        if len(judge_response.msgs) == 0:
            raise RuntimeError("Got no judgement message.")
        judge_response_msg = judge_response.msgs[0]

        if judge_response.terminated:
            raise RuntimeError("Task judgement failed.")

        return TextPrompt(judge_response_msg.content)


class RouteAgent(ChatAgent):
    r"""An agent that dispatches the User Agent's instructions according to 
    the given descriptions or knowledge about the executive agents.

    Args:
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.EB_4`)
        task_type (TaskType, optional): The type of task for which to generate
            a prompt. (default: :obj:`TaskType.TRAVEL_ASSISTANT`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        assistant_model_list (List[Any], optional): The list of models for
            executive agents. (default: :obj:`[]`)
        router_prompt (Union[str, TextPrompt], optional): The prompt for
            routing the instruction. (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agent. (default: :obj:`None`)
    """
    def __init__(
        self,
        model: Optional[ModelType] = None,
        task_type: TaskType = TaskType.TRAVEL_ASSISTANT,
        model_config: Optional[Any] = None,
        assistant_model_list: List[Any] = [],
        router_prompt: Optional[Union[str, TextPrompt]] = None,
        output_language: Optional[str] = None,
    ) -> None:
        self.router_prompt: Union[str, TextPrompt]
        if router_prompt is None:
            router_prompt_template = \
                PromptTemplateGenerator().get_prompt_from_key(task_type, "router_proompt")
            self.router_prompt = format_assistant_agent_list(assistant_model_list, 
                                                            router_prompt_template, 
                                                            output_language)
        else:
            self.router_prompt = TextPrompt(router_prompt)

        model_config = model_config or ErnieBotConfig(temperature=0.05)

        if output_language.lower() in 'chinese' or 'chinese' in output_language.lower():
            sys_prompt = "请选择一个适合执行指令的agent。"
        else:
            sys_prompt = "You can select an executive agent."
        
        system_message = BaseMessage(
            role_name="Instruction Router",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=sys_prompt        
        )

        super().__init__(system_message, 
                         model=model,
                         model_config=model_config,
                         with_declare_output_language=False)

    def run(self,
        instruction: str,
        params: str = None,
    ) -> BaseMessage:
        r"""Route one specific instruction to a related agent.

        Args:
            instruction (str): The insruction to be routed.
            params (str): The input paramters in str format.

        Returns:
            BaseMessage: The response returned by the route agent.
        """
        self.reset()
        route_prompt = self.router_prompt.format(instruction=instruction, 
                                                input=params)

        task_msg = BaseMessage.make_user_message(role_name="Instruction Router",
                                                 content=route_prompt)
        route_response = self.step(task_msg)
        if len(route_response.msgs) == 0:
            raise RuntimeError("Got no route result message.")
        route_response_msg = route_response.msgs[0]

        if route_response.terminated:
            raise RuntimeError("Instruction routing is failed.")

        return route_response_msg


class SpecifyAgent(ChatAgent):
    r"""An agent that specifies a given task prompt by prompting the user to
    provide more details.

    Attributes:
        DEFAULT_WORD_LIMIT (int): The default word limit for the task prompt.
        task_specify_prompt (TextPrompt): The prompt for specifying the task.

    Args:
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        task_type (TaskType, optional): The type of task for which to generate
            a prompt. (default: :obj:`TaskType.AI_SOCIETY`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        task_specify_prompt (Union[str, TextPrompt], optional): The prompt for
            specifying the task. (default: :obj:`None`)
        word_limit (int, optional): The word limit for the task prompt.
            (default: :obj:`50`)
        output_language (str, optional): The language to be output by the
            agent. (default: :obj:`None`)
    """
    DEFAULT_WORD_LIMIT = 50

    def __init__(
        self,
        model: Optional[ModelType] = None,
        task_type: TaskType = TaskType.TRAVEL_ASSISTANT,
        model_config: Optional[Any] = None,
        task_specify_prompt: Optional[Union[str, TextPrompt]] = None,
        word_limit: int = DEFAULT_WORD_LIMIT,
        output_language: Optional[str] = None,
    ) -> None:
        self.task_specify_prompt: Union[str, TextPrompt]
        if task_specify_prompt is None:
            task_specify_prompt_template = \
                PromptTemplateGenerator().get_task_specify_prompt(task_type)
            self.task_specify_prompt = task_specify_prompt_template.format(
                word_limit=word_limit)
        else:
            self.task_specify_prompt = TextPrompt(task_specify_prompt)

        model_config = model_config or ErnieBotConfig(temperature=0.8)

        if output_language.lower() in 'chinese' or 'chinese' in output_language.lower():
            sys_prompt = "请将任务描述得更加具体。"
        else:
            sys_prompt = "You can make a task more specific."
        
        system_message = BaseMessage(
            role_name="Task Specifier",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=sys_prompt        
        )

        super().__init__(system_message, 
                         model=model,
                         model_config=model_config,
                         output_language=output_language,
                         with_declare_output_language=False)

    def run(
        self,
        task_prompt: Union[str, TextPrompt],
        meta_dict: Optional[Dict[str, Any]] = None,
    ) -> TextPrompt:
        r"""Specify the given task prompt by providing more details.

        Args:
            task_prompt (Union[str, TextPrompt]): The original task
                prompt.
            meta_dict (Dict[str, Any], optional): A dictionary containing
                additional information to include in the prompt.
                (default: :obj:`None`)

        Returns:
            TextPrompt: The specified task prompt.
        """
        self.reset()
        task_specify_prompt = self.task_specify_prompt.format(task=task_prompt)

        if meta_dict is not None:
            task_specify_prompt = task_specify_prompt.format(**meta_dict)

        task_msg = BaseMessage.make_user_message(role_name="Task Specifier",
                                                 content=task_specify_prompt)
        specifier_response = self.step(task_msg)
        if len(specifier_response.msgs) == 0:
            raise RuntimeError("Got no specification message.")
        specified_task_msg = specifier_response.msgs[0]

        if specifier_response.terminated:
            raise RuntimeError("Task specification failed.")

        return TextPrompt(specified_task_msg.content)


class PlannerAgent(ChatAgent):
    r"""An agent that helps divide a task into subtasks based on the input
    task prompt.

    Attributes:
        task_planner_prompt (TextPrompt): A prompt for the agent to divide
            the task into subtasks.

    Args:
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
        agent. (default: :obj:`None`)
    """

    def __init__(
        self,
        model: Optional[ModelType] = None,
        model_config: Optional[Any] = None,
        output_language: Optional[str] = None,
    ) -> None:

        self.task_planner_prompt = TextPrompt(
            "Divide this task into subtasks: {task}. Be concise.")
        system_message = BaseMessage(
            role_name="Task Planner",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You are a helpful task planner.",
        )

        super().__init__(system_message, model, model_config,
                         output_language=output_language)

    def run(
        self,
        task_prompt: Union[str, TextPrompt],
    ) -> TextPrompt:
        r"""Generate subtasks based on the input task prompt.

        Args:
            task_prompt (Union[str, TextPrompt]): The prompt for the task to
                be divided into subtasks.

        Returns:
            TextPrompt: A prompt for the subtasks generated by the agent.
        """
        # TODO: Maybe include roles information.
        self.reset()
        task_planner_prompt = self.task_planner_prompt.format(task=task_prompt)

        task_msg = BaseMessage.make_user_message(role_name="Task Planner",
                                                 content=task_planner_prompt)

        task_response = self.step(task_msg)

        if len(task_response.msgs) == 0:
            raise RuntimeError("Got no task planning message.")
        if task_response.terminated:
            raise RuntimeError("Task planning failed.")

        sub_tasks_msg = task_response.msgs[0]
        return TextPrompt(sub_tasks_msg.content)


class TaskCreationAgent(ChatAgent):
    r"""An agent that helps create new tasks based on the objective
    and last completed task. Compared to :obj:`TaskPlannerAgent`,
    it's still a task planner, but it has more context information
    like last task and incomplete task list. Modified from
    `BabyAGI <https://github.com/yoheinakajima/babyagi>`_.

    Attributes:
        task_creation_prompt (TextPrompt): A prompt for the agent to
            create new tasks.

    Args:
        role_name (str): The role name of the Agent to create the task.
        objective (Union[str, TextPrompt]): The objective of the Agent to
            perform the task.
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agent. (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
        max_task_num (int, optional): The maximum number of planned
            tasks in one round. (default: :obj:3)
    """

    def __init__(
        self,
        role_name: str,
        objective: Union[str, TextPrompt],
        model: Optional[ModelType] = None,
        model_config: Optional[Any] = None,
        output_language: Optional[str] = None,
        message_window_size: Optional[int] = None,
        max_task_num: Optional[int] = 3,
    ) -> None:

        task_creation_prompt = TextPrompt(
            """Create a new task with the following objective: {objective}.
Never forget you are a Task Creator of {role_name}.
You must instruct me based on my expertise and your needs to solve the task.
You should consider past solved tasks and in-progress tasks: {task_list}.
The new created tasks must not overlap with these past tasks.
The result must be a numbered list in the format:

    #. First Task
    #. Second Task
    #. Third Task

You can only give me up to {max_task_num} tasks at a time. \
Each task shoud be concise, concrete and doable for a {role_name}.
You should make task plan and not ask me questions.
If you think no new tasks are needed right now, write "No tasks to add."
Now start to give me new tasks one by one. No more than three tasks.
Be concrete.
""")

        self.task_creation_prompt = task_creation_prompt.format(
            objective=objective, role_name=role_name,
            max_task_num=max_task_num)
        self.objective = objective

        system_message = BaseMessage(
            role_name="Task Creator",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You are a helpful task creator.",
        )

        super().__init__(system_message, model, model_config,
                         output_language=output_language,
                         message_window_size=message_window_size)

    def run(
        self,
        task_list: List[str],
    ) -> List[str]:
        r"""Generate subtasks based on the previous task results and
        incomplete task list.

        Args:
            task_list (List[str]): The completed or in-progress
                tasks which should not overlap with new created tasks.
        Returns:
            List[str]: The new task list generated by the Agent.
        """

        if len(task_list) > 0:
            task_creation_prompt = self.task_creation_prompt.format(
                task_list=task_list)
        else:
            task_creation_prompt = self.task_creation_prompt.format(
                task_list="")

        task_msg = BaseMessage.make_user_message(role_name="Task Creator",
                                                 content=task_creation_prompt)
        task_response = self.step(task_msg)

        if len(task_response.msgs) == 0:
            raise RuntimeError("Got no task creation message.")
        if task_response.terminated:
            raise RuntimeError("Task creation failed.")

        sub_tasks_msg = task_response.msgs[0]
        return get_task_list(sub_tasks_msg.content)


class PrioritizationAgent(ChatAgent):
    r"""An agent that helps re-prioritize the task list and
    returns numbered prioritized list. Modified from
    `BabyAGI <https://github.com/yoheinakajima/babyagi>`_.

    Attributes:
        task_prioritization_prompt (TextPrompt): A prompt for the agent to
            prioritize tasks.

    Args:
        objective (Union[str, TextPrompt]): The objective of the Agent to
            perform the task.
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agent. (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
    """

    def __init__(
        self,
        objective: Union[str, TextPrompt],
        model: Optional[ModelType] = None,
        model_config: Optional[Any] = None,
        output_language: Optional[str] = None,
        message_window_size: Optional[int] = None,
    ) -> None:
        task_prioritization_prompt = TextPrompt(
            """Prioritize the following tasks : {task_list}.
Consider the ultimate objective of you: {objective}.
Tasks should be sorted from highest to lowest priority, where higher-priority \
tasks are those that act as pre-requisites or are more essential for meeting \
the objective. Return one task per line in your response.
Do not remove or modify any tasks.
The result must be a numbered list in the format:

    #. First task
    #. Second task

The entries must be consecutively numbered, starting with 1.
The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list \
with any other output.""")

        self.task_prioritization_prompt = task_prioritization_prompt.format(
            objective=objective)
        self.objective = objective

        system_message = BaseMessage(
            role_name="Task Prioritizer",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You are a helpful task prioritizer.",
        )

        super().__init__(system_message, model, model_config,
                         output_language=output_language,
                         message_window_size=message_window_size)

    def run(
        self,
        task_list: List[str],
    ) -> List[str]:
        r"""Prioritize the task list given the agent objective.

        Args:
            task_list (List[str]): The unprioritized tasks of agent.
        Returns:
            List[str]: The new prioritized task list generated by the Agent.
        """
        task_prioritization_prompt = self.task_prioritization_prompt.format(
            task_list=task_list)

        task_msg = BaseMessage.make_user_message(
            role_name="Task Prioritizer", content=task_prioritization_prompt)

        task_response = self.step(task_msg)

        if len(task_response.msgs) == 0:
            raise RuntimeError("Got no task prioritization message.")
        if task_response.terminated:
            raise RuntimeError("Task prioritization failed.")

        sub_tasks_msg = task_response.msgs[0]
        return get_task_list(sub_tasks_msg.content)
