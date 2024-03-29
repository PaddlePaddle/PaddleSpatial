# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re
from loguru import logger

from arche.agents import (
    ChatAgent,
    AssistantAgent,
    CriticAgent,
    PlannerAgent,
    SpecifyAgent, 
    JudgeAgent,
    RouteAgent,
)
from arche.agents.chat_agent import ChatAgentResponse
from arche.agents.task_agent_arche import format_assistant_agent_list
from arche.generators import SystemMessageGenerator
from arche.human import Human
from arche.messages import BaseMessage
from arche.prompts import TextPrompt
from arche.typing import ModelType, RoleType, TaskType
from arche.utils import cast_to_model_type


class ArcheActPlaying:
    r"""Role playing between multiple user-agents and 'multiple' assistant-agents.

    Args:
        assistant_role_name (str): The name of the role played by the
            assistant.
        user_role_name (str): The name of the role played by the user.
        critic_role_name (str): The name of the role played by the critic.
            Role name with :obj:`"human"` will set critic as a :obj:`Human`
            agent, else will create a :obj:`CriticAgent`.
            (default: :obj:`"critic"`)
        task_prompt (str, optional): A prompt for the task to be performed.
            (default: :obj:`""`)
        with_task_specify (bool, optional): Whether to use a task specify
            agent. (default: :obj:`True`)
        with_task_planner (bool, optional): Whether to use a task planner
            agent. (default: :obj:`False`)
        with_critic_in_the_loop (bool, optional): Whether to include a critic
            in the loop. (default: :obj:`False`)
        critic_criteria (str, optional): Critic criteria for the critic agent.
            If not specified, set the criteria to improve task performance.
        model_type (ModelType, optional): Model type that will be used for
            role playing. If specified, it will override the model in all
            agents. (default: :obj:`None`)
        assistant_model_list (List, optional): A list of models to be used
            by the assistant agent. (default: :obj:`[]`)
        with_router (bool, optional): Whether to use a router agent.
        task_type (TaskType, optional): The type of task to perform.
            (default: :obj:`TaskType.AI_SOCIETY`)
        assistant_agent_kwargs (Dict, optional): Additional arguments to pass
            to the assistant agent. (default: :obj:`None`)
        user_agent_kwargs (Dict, optional): Additional arguments to pass to
            the user agent. (default: :obj:`None`)
        route_agent_kwargs (Dict, optional): Additional arguments to pass to
            the route agent. (default: :obj:`None`)
        task_specify_agent_kwargs (Dict, optional): Additional arguments to
            pass to the task specify agent. (default: :obj:`None`)
        task_planner_agent_kwargs (Dict, optional): Additional arguments to
            pass to the task planner agent. (default: :obj:`None`)
        critic_kwargs (Dict, optional): Additional arguments to pass to the
            critic. (default: :obj:`None`)
        sys_msg_generator_kwargs (Dict, optional): Additional arguments to
            pass to the system message generator. (default: :obj:`None`)
        extend_sys_msg_meta_dicts (List[Dict], optional): A list of dicts to
            extend the system message meta dicts with. (default: :obj:`None`)
        extend_task_specify_meta_dict (Dict, optional): A dict to extend the
            task specify meta dict with. (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agents. (default: :obj:`None`)
    """
    def __init__(
        self,
        assistant_role_name: str,
        user_role_name: str,
        *,
        critic_role_name: str = "critic",
        task_prompt: str = "",
        with_task_specify: bool = True,
        skip_init_agents: bool = False,
        with_task_planner: bool = False,
        with_critic_in_the_loop: bool = False,
        critic_criteria: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        assistant_model_list: List[Any] = [],
        with_router: bool = False,
        task_type: TaskType = TaskType.TRAVEL_ASSISTANT,
        assistant_agent_kwargs: Optional[Dict] = None,
        user_agent_kwargs: Optional[Dict] = None,
        route_agent_kwargs: Optional[Dict] = None,
        task_specify_agent_kwargs: Optional[Dict] = None,
        task_planner_agent_kwargs: Optional[Dict] = None,
        critic_kwargs: Optional[Dict] = None,
        sys_msg_generator_kwargs: Optional[Dict] = None,
        extend_sys_msg_meta_dicts: Optional[List[Dict]] = None,
        extend_task_specify_meta_dict: Optional[Dict] = None,
        output_language: Optional[str] = None,
    ) -> None:
        self.with_task_specify = with_task_specify
        self.with_task_planner = with_task_planner
        self.with_critic_in_the_loop = with_critic_in_the_loop
        self.model_type = model_type
        self.assistant_model_list = assistant_model_list
        self.with_router = with_router
        self.task_type = task_type
        self.task_prompt = task_prompt
        self.output_language = output_language

        self.specified_task_prompt: Optional[TextPrompt] = None
        self.task_is_clarified: bool = False
        self.init_specifier(assistant_role_name, user_role_name,
                            task_specify_agent_kwargs,
                            extend_task_specify_meta_dict,
                            output_language)
        
        if self.task_is_clarified and not skip_init_agents:
            self.planned_task_prompt: Optional[TextPrompt] = None
            self.init_planner(task_planner_agent_kwargs,
                              output_language)

            sys_msg_generator = SystemMessageGenerator(
                task_type=self.task_type, **(sys_msg_generator_kwargs or {}))

            (init_assistant_sys_msg, init_user_sys_msg,
            sys_msg_meta_dicts) = self.get_sys_message_info(
                assistant_role_name, user_role_name, sys_msg_generator,
                extend_sys_msg_meta_dicts)

            self.assistant_agent: AssistantAgent
            self.user_agent: ChatAgent
            self.assistant_sys_msg: BaseMessage
            self.user_sys_msg: BaseMessage
            self.route_agent: RouteAgent
            self.init_agents(
                init_assistant_sys_msg,
                assistant_agent_kwargs,
                init_user_sys_msg,
                user_agent_kwargs,
                route_agent_kwargs,
                output_language,
            )
            self.critic: Optional[Union[CriticAgent, Human]] = None
            self.critic_sys_msg: Optional[BaseMessage] = None
            self.init_critic(critic_role_name, critic_criteria, critic_kwargs,
                            sys_msg_generator, sys_msg_meta_dicts)

    def init_specifier(self, 
            assistant_role_name: str, user_role_name: str,
            task_specify_agent_kwargs: Optional[Dict],
            extend_task_specify_meta_dict: Optional[Dict],
            output_language: Optional[str]):
        r"""Initialize a task specify agent to clarify the input user query.
        The outcome specified user query will be used to replace the original
        user query (in the format of task prompt). 
        If there is no task specify agent, specified task prompt will not be generated.

        Args:
            assistant_role_name (str): The name of the role played by the
                assistant.
            user_role_name (str): The name of the role played by the user.
            task_specify_agent_kwargs (Dict, optional): Additional arguments
                to pass to the task specify agent.
            extend_task_specify_meta_dict (Dict, optional): A dict to extend
                the task specify meta dict with.
            output_language (str, optional): The language to be output by the
                agents.
        """
        if self.with_task_specify:
            task_specify_meta_dict = dict()
            if self.task_type in [TaskType.TRAVEL_ASSISTANT]:
                task_specify_meta_dict.update(
                    dict(
                        assistant_role=assistant_role_name, 
                        user_role=user_role_name
                    )
                )
            task_specify_meta_dict.update(extend_task_specify_meta_dict or {})
            if self.model_type is not None:
                if task_specify_agent_kwargs is None:
                    task_specify_agent_kwargs = {}
                task_specify_agent_kwargs.update(dict(model=self.model_type))
            specify_agent = SpecifyAgent(
                task_type=self.task_type,
                output_language=output_language,
                **(task_specify_agent_kwargs or {}),
            )
            self.specified_task_prompt = specify_agent.run(
                self.task_prompt,
                meta_dict=task_specify_meta_dict,
            )
            self.task_prompt = self.specified_task_prompt
            self.task_is_clarified = self.clarify_task()
            if self.task_is_clarified:
                # Get rid of the clarifcation part
                self.task_prompt = self.task_prompt.split('任务描述：')[-1].strip('.。')
    
    def clarify_task(self):
        r"""Check if the task prompt is clarified.
        Returns:
            bool: True if the task prompt is clarified, False otherwise.
        """
        if "需要澄清：True" not in self.specified_task_prompt:
            return True
        if not self.judgement_assess(
            response=self.specified_task_prompt,
            statement="需要澄清",
            output_language=self.output_language
        ):
            return True
        return False

    def judgement_assess(self, 
            response: Union[str, TextPrompt], 
            statement: Union[str, TextPrompt], 
            output_language: Optional[str], 
            judge_agent_kwargs: Optional[Dict] = None) -> bool:
        r"""Initialize a judge agent to check the statement regarding the response 
        is True or False.

        Args:
            response (str): The response to be assessed.
            statement (str): The statement towrads the response. 
            output_language (str, optional): The language to be output by the
                agent.
            judge_agent_kwargs (Dict, optional): Additional arguments
                to pass to the judge agent.
        """
        if self.model_type is not None:
            if judge_agent_kwargs is None:
                judge_agent_kwargs = {}
            judge_agent_kwargs.update(dict(model=self.model_type))
        judge_agent = JudgeAgent(
            task_type=self.task_type,
            output_language=output_language,
            **(judge_agent_kwargs or {}),
        )
        judgement = judge_agent.run(
            source_prompt=response, 
            statement_prompt=statement
        )
        judgement = judgement.strip().lower()
        if judgement == "true":
            status = True
        elif judgement == "false":
            status = False
        else:
            if "true" in judgement.strip().lower():
                status = True
            elif "false" in judgement.strip().lower():
                status = False
            else:
                raise ValueError("Judgement {} is not recognized.".format(judgement))
        return status

    def task_is_done(self, response: str, strict_constraint=False) -> bool:
        r"""Check if the task is done.
        Args:
            response (str): The response to the task prompt.
            strict_constraint (bool, optional): If True, the vigid check condition will be adopted
        Returns:
            bool: True if the task is done, False otherwise.
        """
        if strict_constraint:
            if "<TASK_DONE>" == response:
                return True
            else:
                return False
        if self.assistant_agent.output_language.lower() in "chinese" \
                or "chinese" in self.assistant_agent.output_language.lower():
            done_stmt = "任务：\n\"{}\"\n以上任务已完成".format(self.task_prompt)
        else:
            done_stmt = "Task: \n\"{}\"\nThe above task is done".format(self.task_prompt)
        try:
            if self.judgement_assess(
                response=response,
                statement=done_stmt,
                output_language=self.output_language
            ):
                return True
        except ValueError as ex:
            logger.warning(ex)
        return False

    def get_sys_message_info(
        self,
        assistant_role_name: str,
        user_role_name: str,
        sys_msg_generator: SystemMessageGenerator,
        extend_sys_msg_meta_dicts: Optional[List[Dict]] = None,
    ) -> Tuple[BaseMessage, BaseMessage, List[Dict]]:
        r"""Get initial assistant and user system message with a list of
        system message meta dicts.

        Args:
            assistant_role_name (str): The name of the role played by the
                assistant.
            user_role_name (str): The name of the role played by the user.
            sys_msg_generator (SystemMessageGenerator): A system message
                generator for agents.
            extend_sys_msg_meta_dicts (List[Dict], optional): A list of dicts
                to extend the system message meta dicts with.

        Returns:
            A tuple containing a `BaseMessage` representing the assistant's
            initial system message, a `BaseMessage` representing the user's
            initial system message, and a list of system message meta dicts.
        """
        sys_msg_meta_dicts = [dict(task=self.task_prompt) for _ in range(2)]
        if (extend_sys_msg_meta_dicts is None and self.task_type in [
                TaskType.TRAVEL_ASSISTANT,
        ]):
            extend_sys_msg_meta_dicts = [
                dict(assistant_role=assistant_role_name,
                     user_role=user_role_name) for _ in range(2)
            ]
        # logger.debug(f"sys_msg_meta_dicts: {sys_msg_meta_dicts}")
        # logger.debug(f"extend_sys_msg_meta_dicts: {extend_sys_msg_meta_dicts}")

        if extend_sys_msg_meta_dicts is not None:
            sys_msg_meta_dicts = [{
                **sys_msg_meta_dict,
                **extend_sys_msg_meta_dict
            } for sys_msg_meta_dict, extend_sys_msg_meta_dict in zip(
                sys_msg_meta_dicts, extend_sys_msg_meta_dicts)]
        # logger.debug(f"sys_msg_meta_dicts: {sys_msg_meta_dicts}")

        init_assistant_sys_msg, init_user_sys_msg = (
            sys_msg_generator.from_dicts(
                meta_dicts=sys_msg_meta_dicts,
                role_tuples=[
                    (assistant_role_name, RoleType.ASSISTANT),
                    (user_role_name, RoleType.USER if self.with_router else RoleType.USER_PLAN),
                ],
            ))
        # logger.info(f"user_role: {RoleType.USER if self.with_router else RoleType.USER_PLAN}")
        # logger.info(f"init_assistant_sys_msg: {init_assistant_sys_msg}")
        # logger.info(f"init_user_sys_msg: {init_user_sys_msg}")
        return init_assistant_sys_msg, init_user_sys_msg, sys_msg_meta_dicts

    def init_planner(self,
                    task_planner_agent_kwargs: Optional[Dict],
                    output_language: Optional[str]):
        r"""Initialize a task plan agent. 
        The planned task prompt is generated based on the task
        prompt, which can be original task prompt or specified task prompt
        if available. 

        Args:
            task_planner_agent_kwargs (Dict, optional): Additional arguments
                to pass to the task planner agent.
            output_language (str, optional): The language to be output by the
                agents.
        """
        if self.with_task_planner:
            if self.model_type is not None:
                if task_planner_agent_kwargs is None:
                    task_planner_agent_kwargs = {}
                task_planner_agent_kwargs.update(dict(model=self.model_type))
            task_planner_agent = PlannerAgent(
                output_language=output_language,
                **(task_planner_agent_kwargs or {}),
            )
            self.planned_task_prompt = task_planner_agent.run(self.task_prompt)
            self.task_prompt = (f"{self.task_prompt}\n"
                                f"{self.planned_task_prompt}")
        else:
            self.planned_task_prompt = None

    def init_router(self, 
            route_agent_kwargs: Optional[Dict],
            output_language: Optional[str],
        ) -> None:
        r"""A dedicated agent to route the user message to a specific assistant agent.

        Args:
            route_agent_kwargs (Dict, optional): Additional arguments to
                pass to the router agent.
            output_language (str, optional): The language to be output by the
                agents.
        """
        if self.with_router:
            router_meta_dict = dict()
            if self.task_type in [TaskType.TRAVEL_ASSISTANT]:
                router_meta_dict.update(
                    dict(
                        assistant_model_list=self.assistant_model_list
                    )
                )
            if route_agent_kwargs is None:
                route_agent_kwargs = {}
            if self.model_type is not None:
                route_agent_kwargs.update(dict(model=self.model_type))
            route_agent_kwargs.update(router_meta_dict)
            self.route_agent = RouteAgent(
                task_type=self.task_type,
                output_language=output_language,
                **(route_agent_kwargs or {}),
            )

    def init_agents(
        self,
        init_assistant_sys_msg: BaseMessage,
        assistant_agent_kwargs: Optional[Dict],
        init_user_sys_msg: BaseMessage,
        user_agent_kwargs: Optional[Dict],
        route_agent_kwargs: Optional[Dict],
        output_language: Optional[str],
    ):
        r"""Initialize assistant and user agents with their system messages.

        Args:
            init_assistant_sys_msg (BaseMessage): Assistant agent's initial
                system message.
            assistant_agent_kwargs (Dict, optional): Additional arguments to
                pass to the assistant agent.
            init_user_sys_msg (BaseMessage): User agent's initial system
                message.
            user_agent_kwargs (Dict, optional): Additional arguments to
                pass to the user agent.
            route_agent_kwargs (Dict, optional): Additional arguments to
                pass to the route agent.
            output_language (str, optional): The language to be output by the
                agents.
        """
        if self.model_type is not None:
            # if assistant_agent_kwargs is None:
            #     assistant_agent_kwargs = {}
            # assistant_agent_kwargs.update(dict(model=self.model_type))
            if user_agent_kwargs is None:
                user_agent_kwargs = {}
            user_agent_kwargs.update(dict(model=self.model_type))

        if self.assistant_model_list:
            if assistant_agent_kwargs is None:
                assistant_agent_kwargs = {}
            assistant_agent_kwargs.update(dict(assistant_model_list=self.assistant_model_list))
            if not self.with_router:
                init_user_sys_msg_upd_content = format_assistant_agent_list(
                                                    self.assistant_model_list, 
                                                    init_user_sys_msg.content, 
                                                    output_language)
                init_user_sys_msg = init_user_sys_msg.create_new_instance(content=init_user_sys_msg_upd_content)
                # print(f'\ninit_user_sys_msg: {init_user_sys_msg.content}\n')
            else:
                self.init_router(route_agent_kwargs=route_agent_kwargs, output_language=output_language)

        self.assistant_agent = AssistantAgent(
            init_assistant_sys_msg,
            output_language=output_language,
            **(assistant_agent_kwargs or {}),
        )
        self.assistant_sys_msg = self.assistant_agent.system_message

        self.user_agent = ChatAgent(
            init_user_sys_msg,
            output_language=output_language,
            **(user_agent_kwargs or {}),
        )
        self.user_sys_msg = self.user_agent.system_message

    def init_critic(self, critic_role_name: str,
                    critic_criteria: Optional[str],
                    critic_kwargs: Optional[Dict],
                    sys_msg_generator: SystemMessageGenerator,
                    sys_msg_meta_dicts: List[Dict]):
        r"""Initialize critic agent. If critic role name is :obj:`"human"`,
        create a :obj:`Human` critic agent. Else, create a :obj:`CriticAgent`
        critic agent with specified critic criteria. If the critic criteria
        is not specified, set it to improve task performance.

        Args:
            critic_role_name (str): The name of the role played by the critic.
            critic_criteria (str, optional): Critic criteria for the
                critic agent. If not specified, set the criteria to
                improve task performance.
            critic_kwargs (Dict, optional): Additional arguments to
                pass to the critic.
            sys_msg_generator (SystemMessageGenerator): A system message
                generator for agents.
            sys_msg_meta_dicts (list): A list of system message meta dicts.
        """
        if self.with_critic_in_the_loop:
            if critic_role_name.lower() == "human":
                self.critic = Human(**(critic_kwargs or {}))
            else:
                critic_criteria = (critic_criteria
                                   or "improving the task performance")
                critic_msg_meta_dict = dict(critic_role=critic_role_name,
                                            criteria=critic_criteria,
                                            **sys_msg_meta_dicts[0])
                self.critic_sys_msg = sys_msg_generator.from_dict(
                    critic_msg_meta_dict,
                    role_tuple=(critic_role_name, RoleType.CRITIC),
                )
                if self.model_type is not None:
                    if critic_kwargs is None:
                        critic_kwargs = {}
                    critic_kwargs.update(dict(model=self.model_type))
                self.critic = CriticAgent(
                    self.critic_sys_msg,
                    **(critic_kwargs or {}),
                )

    def init_chat(self) -> Tuple[BaseMessage, List[BaseMessage]]:
        r"""Initializes the chat by resetting both of the assistant and user
        agents, and sending the system messages again to the agents using
        chat messages. Returns the assistant's introductory message and the
        user's response messages.

        Returns:
            A tuple containing a `BaseMessage` representing the assistant's
            introductory message, and a list of `BaseMessage` representing
            the user's response messages.
        """
        self.assistant_agent.reset()
        self.user_agent.reset()

        # Send the system messages again to the agents using chat messages
        user_msg_content = self.user_sys_msg.content.strip()
        user_msg_content = \
            user_msg_content if user_msg_content.endswith(".") or user_msg_content.endswith("。") \
                             else user_msg_content + '.'
        
        if self.assistant_agent.output_language.lower() in "chinese" \
                or "chinese" in self.assistant_agent.output_language.lower():
            assistant_msg_content = (f"{user_msg_content} "
                        "现在请开始给我指示，一条条告诉我。 "
                        "只可以回复指令和输入。")
        else:
            assistant_msg_content = (f"{user_msg_content} "
                        "Now start to give me instructions one by one. "
                        "Only reply with Instruction and Input.")

        assistant_msg = BaseMessage.make_assistant_message(
            role_name=self.assistant_sys_msg.role_name,
            content=assistant_msg_content)
        
        user_msg = BaseMessage.make_user_message(
            role_name=self.user_sys_msg.role_name,
            content=f"{self.assistant_sys_msg.content}")
        
        assistant_response = self.assistant_agent.step(user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            raise ValueError(f"Assistant agent terminated unexpectedly. "
                             f"Error info: {assistant_response.info}")
        
        assistant_res_msg = self.reduce_message_options(assistant_response.msgs)
        # assistant_res_msg.content = f""
        self.assistant_agent.submit_message(assistant_res_msg)

        return assistant_msg, assistant_response.msgs

    @logger.catch
    def reduce_message_options(self,
        messages: Sequence[BaseMessage],
    ) -> BaseMessage:
        r"""Processes a sequence of chat messages, returning the processed
        message. If multiple messages are provided and
        `with_critic_in_the_loop` is `False`, raises a `ValueError`.
        If no messages are provided, a `ValueError` will be raised.

        Args:
            messages: A sequence of `BaseMessage` objects to process.

        Returns:
            A single `BaseMessage` representing the processed message.
        """
        if len(messages) == 0:
            raise ValueError("No messages to process.")
        if len(messages) > 1 and not self.with_critic_in_the_loop:
            raise ValueError("Got than one message to process. "
                             f"Num of messages: {len(messages)}.")
        elif self.with_critic_in_the_loop and self.critic is not None:
            critic_response = self.critic.reduce_step(messages)
            processed_msg = critic_response.msg
        else:
            processed_msg = messages[0]

        return processed_msg

    def search_exe_agent_from_response(self, 
        user_response: BaseMessage
    ) -> Tuple[str, str]:
        r"""Search the agent from the response message if possible and update the user response.
        Args:
            user_response: A `BaseMessage` represents the response message from the user agent.
        Returns:
            A tuple containing the updated user response and extracted executor.
        """
        executor: str = None
        if self.output_language.lower() in 'chinese' or 'chinese' in self.output_language.lower():
            ptn = r"执行者\：\s*(.+)\n*"
        else:
            ptn = r"Executor\:\s*(.+)\n*"
        rsp_content = user_response.content
        match = re.search(ptn, rsp_content)
        if match:
            executor = match.group(1).strip()
            if self.with_router:
                executor_pair = match.group(0)
                rsp_content = rsp_content.split(executor_pair)[0]
                user_response = user_response.create_new_instance(content=rsp_content)
        return user_response, executor

    def extract_keys_from_user_instruction(self, 
        user_msg: BaseMessage) -> Dict:
        r"""Extract the keys from user instruction.
        Args:
        Returns:
        """
        matched_values = dict()
        if self.output_language.lower() in 'chinese' or 'chinese' in self.output_language.lower():
            ptns = [r"指令\：\s*(.+)\n*", r"输入\：\s*(.+)\n*"]
        else:
            ptns = [r"Instruction\:\s*(.+)\n*", r"Input\:\s*(.+)\n*"]
        for i, ptn in enumerate(ptns):
            user_instruction_msg = user_msg.content
            match = re.search(ptn, user_instruction_msg)
            if match:
                value = match.group(1).strip()
                if i == 0:
                    matched_values.update({'instruction': value})
                if i == 1:
                    matched_values.update({'input': value})
        return matched_values

    def router_agent(self, 
            instruction: str, input_params: str
        ) -> str:
        r"""A dedicated agent to route the user message to a specific assistant agent.
        Args:
        Returns:
        """
        rsp_msg = self.route_agent.run(instruction=instruction, params=input_params)
        response_msg, executive_agent = self.search_exe_agent_from_response(rsp_msg)
        return response_msg, executive_agent

    @logger.catch
    def route_assistant_agent(self, 
        user_instruction: BaseMessage
    ) -> ModelType:
        r"""Dispatch and route the input user instruction to a specific assistant agent.
        Args:
        Returns:
        """
        executor: str = None
        executor_model: ModelType = self.assistant_agent.model
        try:
            if not self.with_router:
                user_instruction, executor = self.search_exe_agent_from_response(user_instruction)
            else:
                valued_elems = self.extract_keys_from_user_instruction(user_msg=user_instruction)
                assert len(valued_elems) > 1, \
                    f"At lease one of Two keys ('Instruction' and 'Input') are missed {valued_elems}."
                instruction, input_params = None, None
                if 'instruction' not in valued_elems:
                    raise RuntimeError(f"CAN NOT route an instruction w/o 'instruction': {valued_elems}")
                instruction = valued_elems['instruction']
                if 'input' in valued_elems:
                    input_params = valued_elems['input']
                response_msg, executor = self.router_agent(instruction=instruction, input_params=input_params)

                if self.output_language.lower() in 'chinese' or 'chinese' in self.output_language.lower():
                    user_instruction += "\n" + "执行者：" + executor
                else:
                    user_instruction += "\n" + "Executor: " + executor

                if response_msg.content.startswith("\n"):
                    user_instruction += response_msg.content
                else:
                    user_instruction += "\n" + response_msg.content
            if executor:
                executor_model = cast_to_model_type(executor)
            logger.info(f"executor: {executor}")
            logger.info(f"executor_model: {executor_model}")
            logger.info(f"user_instruction: {user_instruction}")
        except ValueError as ex:
            logger.error(f'\nFailed to find executor from response: {ex}\n')
            return user_instruction, executor_model
        except (AssertionError, RuntimeError) as ex:
            logger.error("User instruction is twisted.")
            user_instruction = user_instruction.create_new_instance(content="<TASK_DONE>")
        return user_instruction, executor_model

    def step(
        self,
        assistant_msg: BaseMessage,
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        r"""Advances the conversation by taking a message from the assistant,
        processing it using the user agent, and then processing the resulting
        message using the assistant agent. Returns a tuple containing the
        resulting assistant message, whether the assistant agent terminated
        the conversation, and any additional assistant information, as well as
        a tuple containing the resulting user message, whether the user agent
        terminated the conversation, and any additional user information.

        Args:
            assistant_msg: A `BaseMessage` representing the message from the
                assistant.

        Returns:
            A tuple containing two ChatAgentResponse: the first struct contains
            the resulting assistant message, whether the assistant agent
            terminated the conversation, and any additional assistant
            information; the second struct contains the resulting user message,
            whether the user agent terminated the conversation, and any
            additional user information.
        """
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (ChatAgentResponse([], False, {}),
                    ChatAgentResponse([], user_response.terminated,
                                      user_response.info))
        user_msg = self.reduce_message_options(user_response.msgs)
        self.user_agent.submit_message(user_msg)
        # TODO: route the user_msg with the router agent (if exists)
        if self.task_is_done(user_msg.content, strict_constraint=True):
            return (
                ChatAgentResponse([], False, {}),
                ChatAgentResponse([user_msg], user_response.terminated, user_response.info),
            )
        routed_user_msg, exec_model = self.route_assistant_agent(user_instruction=user_msg)

        assistant_response = self.assistant_agent.step(routed_user_msg, exec_model=exec_model)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (ChatAgentResponse([], assistant_response.terminated,
                                      assistant_response.info),
                    ChatAgentResponse([user_msg], False, user_response.info))
        assistant_msg = self.reduce_message_options(assistant_response.msgs)
        self.assistant_agent.submit_message(assistant_msg)

        return (
            ChatAgentResponse([assistant_msg], assistant_response.terminated,
                              assistant_response.info),
            ChatAgentResponse([routed_user_msg], user_response.terminated,
                              user_response.info),
        )


