# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc:
Authors: 
Date: 
"""
from typing import Any

from arche.prompts import TextPrompt, TextPromptDict
from arche.typing import RoleType


# flake8: noqa :E501
class TravelAssistantPromptTemplateDict(TextPromptDict):
    r"""A dictionary containing :obj:`TextPrompt` used in the `AI Native Travel Assistant`
    task.

    Attributes:
        TASK_SPECIFY_PROMPT (TextPrompt): A prompt to specify a task in more
            detail.
        ASSISTANT_PROMPT (TextPrompt): A system prompt for the AI assistant
            that outlines the rules of the conversation and provides
            instructions for completing tasks.
        USER_PROMPT (TextPrompt): A system prompt for the AI user that
            outlines the rules of the conversation and provides instructions
            for giving instructions to the AI assistant.
    """

    OPERATOR_DICT = {
        "poi_find":
            ("The most basic query function of the map, used to retrieve real POI locations on the map app. "
             "By default, it prefers to return points that are closer. Specific POIs can be searched by name, "
             "and specific categories of POIs can be searched by category. When the name or category is not clear, "
             "the POI can also be searched by describing its features."),
        "navi_route_find":
            ("Given necessary information such as starting point, destination, and transportation mode, carry out route planning. "
             "The starting point and destination need to be correctly extracted from the query. Except for 'home', "
             "'company', 'current location', all starting points, destinations, "
             "and waypoints need to be queried in advance through the poi_find function."),
        "cab_call":
            ("Given necessary information such as starting point and destination, carry out a ride-hailing call. "
             "Except for 'home', 'company', 'current location', all starting and ending points, waypoints, etc., "
             "need to be queried in advance through the poi_find function."),
        "pub_trans":
            ("After giving parameters such as bus routes and station locations, query public transport route information. "
             "Public transport routes include various transportation nodes such as buses and subways. The target_station is "
             "determined by both the route and station location, with the station location defaulting to "
             "'current location station'."),
        "ask_info":
            ("Aimed at questions users want to know, asking for specific information about one or more targets. "
             "The target can be a POI or route. "
             "For a POI, the content of the questions, obj_ask, includes but not limited to business hours, phone number, "
             "travel strategies, etc. By default, obj_ask='introduction' can be used to show the basic information "
             "or introduction of a POI. "
             "For a route, the content of the questions, obj_ask, includes but not limited to the length of the journey, "
             "duration, tolls, waiting time, estimated fare, etc."
            ), 
        "map_func_call":
            ("Map function instructions, including but not limited to playing music, querying current location, "
             "increasing volume, viewing the destination of the route, querying traffic restriction information, "
             "navigation instructions, etc."),
        "ref_LLM":
            ("When a user's query exceeds the service scope of the map app, a general language model can be called "
             "to give a general answer to the user.")
    }
    OPERATOR_DESCS = '\n'.join([': '.join([k, v]) for k, v in OPERATOR_DICT.items()])

    CLEAR_EXAMPLES = ("""

Here are some examples that don't require clarification:

Query: What tourist attractions are there?
Analysis: The user's intention is [poi_find], [location] defaults to 'current location', [category] is 'attraction', """
"""the intent is clear, so no clarification is needed.
Need clarification: False
Task description: Search for POIs nearby the current location, categorized as tourist attraction

Query: Camping and study tours
Analysis: The user's intention is [poi_find], [location] defaults to 'current location', [feature] is 'camping and study """
"""tours', the intent is clear, so no clarification is needed.
Need clarification: False
Task description: Search for POIs nearby the current location, featuring camping and study tours

Query: Stay in a hotel, where?
Analysis: The user's intention is [poi_find], [location] defaults to 'current location', [category] is 'hotel', """
"""the intent is clear, so no clarification is needed.
Need clarification: False
Task description: Search for POIs nearby the current location, categorized as hotel
    """)

    CLARIFY_EXAMPLES = """

Here are some examples that require clarification:

Query: What about taking a train?
Analysis: The user's intention is [navi_route_find], [transportation_mode] is 'train', [starting_point] defaults to 'current location', but the [destination] is missing, therefore clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: If I choose the high-speed rail, what are the train numbers?
Analysis: The user's intention is [navi_route_find], [transportation_mode] is 'high-speed rail', [starting_point] defaults to 'current location', but the [destination] is missing, therefore clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: What flights are available?
Analysis: The user's intention is [navi_route_find], [transportation_mode] is 'airplane', [starting_point] defaults to 'current location', but the [destination] is missing, therefore clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: How much does it cost from Beijing to Hangzhou?
Analysis: The user's intention is [navi_route_find] and [ask_info], [starting_point] is 'Beijing', [destination] is 'Hangzhou', but the [transportation_mode] is missing, therefore clarification is needed.
Need clarification: True
Question: May I ask what your transportation mode is?

Query: Check the bus.
Analysis: The user's intention is [pub_trans], [target_station] is unclear, bus station is by default 'current location station', but the bus route is missing, so clarification is needed.
Need clarification: True
Question: What bus route are you looking to inquire about?

Query: How do I get there from Software Park Ark Building?
Analysis: The user's intention is [navi_route_find], [starting_point] is 'Software Park Ark Building', [transportation_mode] is by default 'driving', but the [destination] is missing, so clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: Inside the scenic area, what are the recommended routes?
Analysis: The user's intention is [ask_info], asking about recommended routes in a scenic area, [obj_ask] is 'route recommendation', but the [target_poi] is missing, so clarification is needed.
Need clarification: True
Question: May I ask which scenic area are you asking about?

Query: Departing from Golden Domain Park I in Changping District.
Analysis: The user's intent is [navi_route_find], with the [starting point] being 'Golden Domain Park I in Changping District', and the default [transportation mode] is 'driving', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: What transport is recommended for returning from Beijing?
Analysis: The user's intent is [navi_route_find] and [ask_info], with the [starting_point] being 'Beijing', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: How much for the fastest high-speed train?
Analysis: The user's intent is [navi_route_find] and [ask_info], with the default [starting_point] being 'current location', [transportation_mode] is 'high-speed train', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: Can you initiate a taxi service?
Analysis: The user's intent is to initiate a taxi [cab_call], with the default [starting_point] being 'current location', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: Okay, may I ask where your destination is?

Query: Help me check how to get there.
Analysis: The user's intent is [navi_route_find], with the default [starting_point] being 'current location', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: Travel plan.
Analysis: The user's intent is [navi_route_find], with the default [starting_point] being 'current location', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: Inside the Grand Mansion.
Analysis: The user's intent is unclear, thus clarification is needed.
Need clarification: True
Question: Are you planning to search for a place or ask for directions?

Query: A flight from Jieyang Airport to a city and then back to Beijing.
Analysis: The user's intent is [navi_route_find], with the [starting_point] being 'Jieyang Airport' and the [destination] being 'Beijing', but the [waypoint] is unclear, thus clarification is needed.
Need clarification: True
Question: Which city are you referring to for the stopover?

Query: How to buy tickets.
Analysis: The user's intent is [navi_route_find] and [ask_info], with the default [starting_point] being 'current location', but the [destination] and [transportation_mode] are missing, thus clarification is needed.
Need clarification: True
Question: Where are you going? What is the mode of transportation?

Query: How to get there from Wuhan.
Analysis: The user's intent is [navi_route_find], with the [starting_point] being 'Wuhan', but the [destination] is missing, thus clarification is needed.
Need clarification: True
Question: May I ask where your destination is?

Query: What is there to do there?
Analysis: The user's intent is [poi_find], but the reference to 'there' is unclear, and the [location] is missing, so clarification is needed.
Need clarification: True
Question: May I ask where 'there' is that you are referring to?

Query: How long will it take to walk out?
Analysis: The user's intent is [ask_info], asking about the time duration of a route, but the [target_route] is missing, so clarification is needed.
Need clarification: True
Question: May I ask which route you are inquiring about?

Query: Which day is recommended to go out and play?
Analysis: The user's intent is unclear, possibly [navi_route_find] and [ask_info], but the [destination] is missing, so clarification is needed.
Need clarification: True
Question: Where do you want to go to play?

Query: What should I pay attention to when taking children?
Analysis: The user's intent is [ask_info], with [obj_ask] being 'what should I pay attention to when taking children', but the [target_poi] is missing, so clarification is needed.
Need clarification: True
Question: May I ask which place you are inquiring about?

Query: What about taking a plane?
Analysis: The user's intent might be [ask_info], [transportation_mode] is 'plane', the default [starting_point] is 'current location', but the [destination] is missing, so clarification is needed.
Need clarification: True
Question: Are you inquiring about plane flights? What are the starting and ending points?

Query: Refer to a specific tourist spot.
Analysis: The user's intent is unclear, so clarification is needed.
Need clarification: True
Question: Are you looking to search for a tourist spot?

    """

    TASK_SPECIFY_PROMPT: TextPrompt = TextPrompt(
        ("""This is a task:
'''
{task}
'''

Given the set of functionalities as follows:
<FUNCS_DESC>
{funcs_desc}
</FUNCS_DESC>

If {assistant_role} can support the task: {task}, then the original task description is feasible. """
"""Please ensure that the generated task description is as specific as possible and ensures logicality and accuracy.
Make sure the generated task description is brief, not exceeding {word_limit} words, """
"""and do not add any extra content unrelated to the task.
The output format is:
Query: <USER_QUERY>
Analysis: <REASON>
Need clarification: False
Task description: <SPECIFIED_TASK>
{clear_examples}

If it is determined that the known set of functionalities cannot meet {user_role}'s query, """
"""then the original task description is not feasible, 
and it is necessary to ask questions to {user_role} to guide {user_role} to clarify the original task description.
If there are unclear parts in the task description, be sure to ask clarifying questions to the user.
For tasks that cannot be completed, questions must be asked to {user_role}. 
Do not assume that the user will provide additional input in subsequent interactions.
Ensure that the clarifying questions generated are related to the task, and do not add """
"""any extra content unrelated to the task.
The output format is:
Query: <USER_QUERY>
Analysis: <REASON>
Need clarification: True
Question: <CLARIFIED_QUESTION>
{clarify_examples}
""")
    )

    TASK_SIMPLIFY_PROMPT: TextPrompt = TextPrompt(
        """Conversations:
'''
{task}
'''

Please summarize, specify, and simplify the {user_role}'s (clarified) query according to the above conversions in no more than {word_limit} words. 
DO NOT add any extra content unrelated to the {user_role}'s request.

Please output the summarized {user_role}'s request: 
"""
    )

    JUDGE_PROMPT: TextPrompt = TextPrompt(
        """Please carefully understand the following text:
'''
{description}
'''

And judge this statement:
'''
{statement}
'''
as "True" or "False".

Your answer can only be "True" or "False".

Please give your judgment: """
    )

    ASSISTANT_PROMPT: TextPrompt = TextPrompt(
        """
Please make sure not to forget that you are a {assistant_role}, and I am a {user_role}. 
Please do not reverse the roles! Avoid sending commands to me!
We need to work together to complete a common task: "{task}"
Please help me complete this task: "{task}"
Never forget our task!

The professional skills you can use are as follows:
<FUNCS_DESC>
{funcs_desc}
</FUNCS_DESC>

I should guide you in completing this task based on my needs and your professional knowledge.
I can only give you one instruction at a time.
You must provide an accurate, specific, and executable answer based on the instruction received.
If you cannot execute an instruction received due to physical, moral, legal reasons, or limitations of your own capabilities, you must refuse my instruction and explain why.
You must provide a solution based on the instruction received; beyond this, please do not add any irrelevant content.
You should not ask me questions or send instruction requests.
Your solution should be complete and accurate.
You should describe your solution in a declarative sentence format.
Unless I declare the task completed, the first sentence of your answer should always be in the following form:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and include how it can be implemented.
Please use "Next instruction." as the last sentence of your answer."""
    )

    ROUTER_PROMPT: TextPrompt = TextPrompt(
        """
Based on the instruction: "{instruction}" and input: {input}, choose the appropriate executive agent.
Available executors include: {assistant_agents}. They have different areas of expertise and capabilities:
{assistant_agent_details}

You must select the executor suitable for carrying out the instruction ("{instruction}") based on the different areas of expertise and capabilities of {assistant_agents}.
You can only give your instruction in the following way:

Reason: <YOUR_EXPLANATION>
Executor: <SELECTED_AGENT>

"Reason" must be a further, specific reason and explanation for choosing the "Executor".
"Executor" must be one of {assistant_agents}.

You can only choose one "Executor" at a time.
Now, please start making your choice from {assistant_agents}.

You must give your choice in the format mentioned above.
Your answer can only include "Reason" and "Executor". Beyond this, your answer should not contain any other content. """
    )

    USER_PROMPT: TextPrompt = TextPrompt(
        """
You are a {user_role}, and {assistant_role} will help you complete the task: "{task}"

You need to work closely with {assistant_role} to complete a common task: "{task}"
{assistant_role} will assist you in completing this task. Please do not reverse roles!
Do not forget the task!

You must guide {assistant_role} to complete the task: "{task}"

You should, based on your needs and the professional knowledge and abilities of {assistant_role}, guide them to complete this task.
The professional skills of {assistant_role} are as follows:
<FUNCS_DESC>
{funcs_desc}
</FUNCS_DESC>

You can only request {assistant_role} to perform actions in the following two ways:

1. Instruction including input:
Instruction: <INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruction without input:
Instruction: <INSTRUCTION>
Input: None

"Instruction" must be a specific task or action plan.
"Input" provides further, specific contextual information to describe the "Instruction".

You can only give {assistant_role} one instruction request at a time.
{assistant_role} must provide an accurate, specific solution based on the instruction request received.
If due to physical, moral, legal reasons, or limitations of {assistant_role}'s own capabilities, {assistant_role} cannot execute the instruction request,
{assistant_role} should refuse your instruction request and explain the reasons for refusing to execute your instruction request.
You should guide {assistant_role} not to ask you questions or send instruction requests.
Now, please start issuing instruction requests to {assistant_role}.

You must issue instruction requests to {assistant_role} in the two forms of instructions mentioned above.
Your instruction requests can only include "Instruction" and "Input". Beyond this, your instruction requests should not contain any other content.
Before you consider the task complete, you must keep sending instruction requests to {assistant_role}.
When you believe the task is complete, you can only reply to {assistant_role}: "<TASK_DONE>".
Unless {assistant_role}'s reply has resolved the task: "{task}", do not reply to {assistant_role} with "<TASK_DONE>". """
    )

    USER_PROMPT_PLAN: TextPrompt = TextPrompt(
        """
You are a {user_role}, and {assistant_role} will help you complete the task: "{task}"

You need to collaborate closely with {assistant_role} to complete a common task: "{task}"
{assistant_role} will assist you in completing this task. Please make sure not to reverse roles!
Do not forget this task!

You must decide, based on the different abilities of members within {assistant_agents}, which one is more suitable as the executor for {assistant_role} to complete the task: {task}

{assistant_agents} have different areas of expertise and abilities, so you need to make a choice based on their distinct characteristics.
The professional capabilities of different members in {assistant_agents} are described as follows:
{assistant_agent_details}

You must guide {assistant_role} and choose the appropriate executor to complete the task: "{task}"
Please provide a reason for choosing the executor.

You should guide them to complete this task based on your needs and the professional knowledge and capabilities of {assistant_role}.
You can only request {assistant_role} to perform actions in the following two ways:

1. Instruction including input:
Instruction: <INSTRUCTION>
Input: <YOUR_INPUT>
Reason: <SELECT_AGENT_REASON>
Executor: <SELECTED_AGENT>

2. Instruction without input:
Instruction: <INSTRUCTION>
Input: None
Reason: <SELECT_AGENT_REASON>
Executor: <SELECTED_AGENT>

"Instruction" must be a specific task or action plan.
"Input" provides further, specific contextual information to describe the "Instruction".
"Reason" is the reason for choosing the executor.
"Executor" must be one of the members of {assistant_agents}.

You can only give {assistant_role} one instruction request at a time.
{assistant_role} must provide an accurate, specific solution based on the instruction request received.
If {assistant_role} cannot execute the instruction request due to physical, moral, legal reasons, or limitations of their own capabilities,
{assistant_role} should refuse your instruction request and explain why.
You should guide {assistant_role} not to ask you questions or send instruction requests.
Now, please start issuing instruction requests to {assistant_role}.

Your instruction requests to {assistant_role} must follow the two forms of instructions mentioned above.
Your instruction requests can only include "Instruction", "Executor", "Reason for selecting the executor", and "Input".
Beyond this, your instruction requests should not contain any other content.
Before you consider the task complete, you must keep sending instruction requests to {assistant_role}.
When you believe the task is complete, you can only reply to {assistant_role}: "<TASK_DONE>".
Unless {assistant_role}'s reply has resolved the task: "{task}", do not reply to {assistant_role} with "<TASK_DONE>". """
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update({
            "task_specify_prompt": self.TASK_SPECIFY_PROMPT.format(
                                                            funcs_desc=self.OPERATOR_DESCS, 
                                                            clear_examples=self.CLEAR_EXAMPLES, 
                                                            clarify_examples=self.CLARIFY_EXAMPLES),
            "task_simplify_prompt": self.TASK_SIMPLIFY_PROMPT,
            "judge_prompt": self.JUDGE_PROMPT,
            "router_prompt": self.ROUTER_PROMPT,
            RoleType.ASSISTANT: self.ASSISTANT_PROMPT.format(funcs_desc=self.OPERATOR_DESCS),
            RoleType.USER: self.USER_PROMPT.format(funcs_desc=self.OPERATOR_DESCS),
            RoleType.USER_PLAN: self.USER_PROMPT_PLAN
        })
