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
class TravelAssistantEnPromptTemplateDict(TextPromptDict):
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
        "search_for_poi":
            ("The most basic query function of a map, used for retrieving Point of Interest (POI) "
             "that actually exist on a map app. "
             "By default, it prioritizes returning points that are relatively closer. "
             "Specific POIs can be searched by name, "
             "and particular types of POIs can be searched using place categories. "
             "When the name or category is unclear, "
             "one can also describe the characteristics of the place to find the needed POI."),
        "qa_for_poi":
            ("Targeting the questions users want to understand, it involves asking about specific "
             "information of one or several Points of Interest (POI). The content of the inquiry, 'qa_target', "
             "includes but is not limited to business hours, phone numbers, travel tips, etc. By default, "
             "one can use 'qa_target=introduction' to display the most basic information "
             "or introduction of a POI."),
        "search_for_navi":
            ("Given the necessary information such as the starting point and destination, carry out route planning. "
             "The starting point and destination need to be correctly extracted from the query. Besides 'home', "
             "'company', and 'current location', all starting points, destinations, and waypoints need to be "
             "pre-queried using the 'search_for_poi' function."),
        "qa_for_navi":
            ("Ask about specific details of the route planning. The content of the inquiry includes but is not "
             "limited to the length of the journey, time consumption, road fees, etc."),
        "ride_hailing":
            ("Given necessary information such as the starting point and destination, proceed with calling a taxi. "
             "Apart from 'home', 'company', and 'current location', all starting and ending points, as well as "
             "waypoints, need to be pre-queried using the 'search_for_poi' function"),
        "qa_for_ride_hailing":
            ("Inquire about specific details of the taxi route planning. The content of the inquiry includes but "
             "is not limited to travel time, waiting time, estimated road fees, etc."),
        "realtime_mass_transit":
            ("After providing parameters such as bus routes and station locations, inquire about bus route "
             "information. Bus routes include various modes of transportation like buses and subways. "
             "The target station is determined jointly by the bus route and station location. The default station "
             "location is 'current location station'."),
        "qa_for_realtime_mass_transit":
            ("After querying bus route information using realtime_mass_transit, inquire about real-time bus "
            "information, such as what time the number 29 bus will arrive at the station."),
        "call_map_native_function":
            ("Map function commands, which include but are not limited to playing music, querying current location, "
             "increasing volume, viewing the destination of the route, inquiring about traffic restrictions, "
             "navigation commands, and so on."), 
        "ask_for_LLM":
            ("When a user's query exceeds the service scope of the map app, a general language model can be called "
             "upon to provide a generic response to the user.")
    }
    OPERATOR_DESCS = '\n'.join([': '.join([k, v]) for k, v in OPERATOR_DICT.items()])

    CLARIFY_EXAMPLES = """

Here are the translations of the provided examples:

Query: What tourist attractions are there?
Analysis: The user's intention is [search_for_poi], [location] is by default 'current location', [category] is 'tourist attraction'. The intention is clear, so no clarification is needed.
Needs clarification: False
Task description: Search for POIs categorized as tourist attractions near the current location.

Query: Outdoor camping and study.
Analysis: The user's intention is [search_for_poi], [location] is by default 'current location', [feature] is 'outdoor camping and study'. The intention is clear, so no clarification is needed.
Needs clarification: False
Task description: Search for POIs with features of outdoor camping and study near the current location.

Query: Where should I stay in a hotel?
Analysis: The user's intention is [search_for_poi], [location] is by default 'current location', [category] is 'hotel'. The intention is clear, so no clarification is needed.
Needs clarification: False
Task description: Search for hotels as POIs near the current location.

Query: What about taking a train?
Analysis: The user's intention is [search_for_navi], [transportation mode] is 'train', [starting point] is by default 'current location', but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: If I choose high-speed rail, what are the train numbers?
Analysis: The user's intention is [search_for_navi], [transportation mode] is 'high-speed rail', [starting point] is by default 'current location', but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: What flights are available?
Analysis: The user's intention is [search_for_navi], [transportation mode] is 'plane', [starting point] is by default 'current location', but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: How much does it cost from Beijing to Hangzhou?
Analysis: The user's intention is [search_for_navi] and [qa_for_navi], [starting point] is 'Beijing', [destination] is 'Hangzhou', but the [transportation mode] is missing, so clarification is needed.
Needs clarification: True
Question: What is your mode of transportation?

Query: Check the bus.
Analysis: The user's intention is [realtime_mass_transit], [target station] is unclear, station is by default 'current location station', but the bus route is missing, so clarification is needed.
Needs clarification: True
Question: What bus route are you looking to inquire about?

Query: How do I get there from Software Park Ark Building?
Analysis: The user's intention is [search_for_navi], [starting point] is 'Software Park Ark Building', [transportation mode] is by default 'driving', but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: Inside the scenic area, what are the recommended routes?
Analysis: The user's intention is [qa_for_poi], asking about recommended routes in a scenic area, qa_target is 'route recommendation', but the [poi_target] is missing, so clarification is needed.
Needs clarification: True
Question: Which scenic area are you asking about?

Query: Departing from Golden Domain I in Changping District.
Analysis: The user's intent is [search_for_navi], with the [starting point] being 'Golden Domain I in Changping District', and the default [transportation mode] is 'driving', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: What transport is recommended for returning from Beijing?
Analysis: The user's intent is [search_for_navi] and [qa_for_navi], with the [starting point] being 'Beijing', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: How much for the fastest high-speed train?
Analysis: The user's intent is [search_for_navi] and [qa_for_navi], with the default [starting point] being 'current location', [transportation mode] is 'high-speed train', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: Can you initiate a taxi service?
Analysis: The user's intent is to initiate a taxi [ride_hailing], with the default [starting point] being 'current location', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: Help me check how to get there.
Analysis: The user's intent is [search_for_navi], with the default [starting point] being 'current location', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: Travel plan.
Analysis: The user's intent is [search_for_navi], with the default [starting point] being 'current location', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: Inside the Grand Mansion.
Analysis: The user's intent is unclear, thus clarification is needed.
Needs clarification: True
Question: Are you planning to search for a place or ask for directions?

Query: A flight from Jieyang Airport to a city and then back to Beijing.
Analysis: The user's intent is [search_for_navi], with the [starting point] being 'Jieyang Airport' and the [destination] being 'Beijing', but the [stopover city] is unclear, thus clarification is needed.
Needs clarification: True
Question: Which city are you referring to for the stopover?

Query: How to buy tickets.
Analysis: The user's intent is [search_for_navi] and [qa_for_navi], with the default [starting point] being 'current location', but the [destination] and [transportation mode] are missing, thus clarification is needed.
Needs clarification: True
Question: Where are you going? What is the mode of transportation?

Query: How to get there from Wuhan.
Analysis: The user's intent is [search_for_navi], with the [starting point] being 'Wuhan', but the [destination] is missing, thus clarification is needed.
Needs clarification: True
Question: What is your destination?

Query: What is there to do there?
Analysis: The user's intent is [search_for_poi], but the reference to 'there' is unclear, and the [location] is missing, so clarification is needed.
Needs clarification: True
Question: What place are you referring to as 'there'?

Query: How long will it take to walk out?
Analysis: The user's intent is [qa_for_navi], asking about the time duration of a route, but the [target route] is missing, so clarification is needed.
Needs clarification: True
Question: What route are you asking about?

Query: Which day is recommended to go out and play?
Analysis: The user's intent is unclear, possibly [search_for_navi] and [qa_for_navi], but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: Where do you want to go to play?

Query: What should I be aware of when going with children?
Analysis: The user's intent is [qa_for_poi], with [qa_target] being 'what to be aware of when going with children', but the [poi_target] is missing, so clarification is needed.
Needs clarification: True
Question: What place are you inquiring about?

Query: What about taking a plane?
Analysis: The user's intent might be [qa_for_navi], [transportation mode] is 'plane', the default [starting point] is 'current location', but the [destination] is missing, so clarification is needed.
Needs clarification: True
Question: Are you inquiring about plane flights? What are the starting and ending points?

Query: Refer to a tourist spot.
Analysis: The user's intent is unclear, so clarification is needed.
Needs clarification: True
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
Needs clarification: False
Task Description: <SPECIFIED_TASK>

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
Needs clarification: True
Question: <CLARIFIED_QUESTION>
{clarify_examples}
""")
    )

    # TODO:
    JUDGE_PROMPT: TextPrompt = TextPrompt(
        """
        """
    )

    # TODO:
    ASSISTANT_PROMPT: TextPrompt = TextPrompt(
        """
Never forget you are a {assistant_role} and I am a {user_role}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a travel task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately solves the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to 
physical, moral, legal reasons or your capability and explain the reasons.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be very specific, provide preferable detailed implementations for task-solving.
Always end <YOUR_SOLUTION> with: Next request."""
    )

    # TODO:
    ROUTER_PROMPT: TextPrompt = TextPrompt(
        """
        """
    )

    # TODO:
    USER_PROMPT: TextPrompt = TextPrompt(
        """
Never forget you are a {user_role}. Never flip roles! You will always ask {assistant_role_list} 
to complete the task: {task}.
You and the {assistant_role_list} share a common interest in collaborating to successfully complete a task.
The {assistant_role_list} will help you to complete the task.

Here is the task: {task}. Never forget the task!
Here are the expertise of {assistant_role_list}:
{assistant_role_list_details}

You must place your query based on {assistant_role_list}'s expertise and your needs 
to solve the task ONLY in the following one way:

1. Claim your query:
Query: <YOUR_QUERY>

You must provide one query at a time.
{assistant_role_list} must decline your query honestly if {assistant_role_list} cannot compelete the query due to 
physical, moral, legal reasons or the capability and explain the reasons.
Now you must start to ask {assistant_role_list} using the one way described above.
Do not add anything else other than your query!
Keep giving the queries until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless the responses have solved your task."""
    )

    # TODO:
    USER_PROMPT_PLAN: TextPrompt = TextPrompt(
        """
Never forget you are a {user_role}. Never flip roles! You will always instruct a {assistant_role}.
You and the {assistant_role} share a common interest in collaborating to successfully complete a task: {task}.
The {assistant_role} will help you to complete the task.
Here is the task: {task}. Never forget the task!
You must instruct the {assistant_role} based on {assistant_role}'s expertise and your needs to 
solve the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information 
for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to 
physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update({
            "task_specify_prompt": self.TASK_SPECIFY_PROMPT.format(
                                                            funcs_desc=self.OPERATOR_DESCS, 
                                                            clarify_examples=self.CLARIFY_EXAMPLES),
            "judge_prompt": self.JUDGE_PROMPT,
            "router_proompt": self.ROUTER_PROMPT,
            RoleType.ASSISTANT: self.ASSISTANT_PROMPT,
            RoleType.USER: self.USER_PROMPT,
            RoleType.USER_PLAN: self.USER_PROMPT_PLAN
        })
