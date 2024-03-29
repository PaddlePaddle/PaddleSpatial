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
        TASK_SPECIFY_PROMPT_CN (TextPrompt): A prompt to specify a task in more
            detail.
        ASSISTANT_PROMPT_CN (TextPrompt): A system prompt for the AI assistant
            that outlines the rules of the conversation and provides
            instructions for completing tasks.
        USER_PROMPT_CN (TextPrompt): A system prompt for the AI user that
            outlines the rules of the conversation and provides instructions
            for giving instructions to the AI assistant.
    """

    OPERATOR_DICT_CN = dict(
        {
            "search_for_poi":
                "地图最基础的查询功能，用于检索地图app上真实存在的POI地点。在默认情况下，会优先返回距离比较近的点。" \
                + "具体的POI可以按名称搜索，特定类别的POI可以使用地点类别搜索，当不明确名称或类别时，也可以只用地点特色描述需要查找的POI。",
            "qa_for_poi":
                "针对用户想要了解的问题，对单个或多个POI点的具体信息进行询问。" \
                + "询问的内容qa_target包括但不限于营业时间、电话、游玩攻略等等。在默认情况下，可以使用qa_target='简介'来展示一个POI的最基本信息或者介绍。",
            "search_for_navi":
                "给定起点、终点等必要信息，进行路线规划。起点，终点需要从query中得到正确的提取。除了'家'、'公司'、'当前位置'之外，所有的起点、终点、" \
                + "途经点等都需要预先经过search_for_poi函数来查询。",
            "qa_for_navi":
                "对路线规划的具体信息进行询问，询问的内容包括但不限于路程长度、耗时、路费等等。",
            "ride_hailing":
                "给定起点、终点等必要信息，进行打车呼叫。" \
                "除了'家'、'公司'、'当前位置'之外，所有的起终点、途经点等都需要预先经过search_for_poi函数来查询。",
            "qa_for_ride_hailing":
                "对打车路线规划的具体信息进行询问，询问的内容包括但不限于路程耗时、等待时间、预估路费等等。",
            "realtime_mass_transit":
                "给定公交线路、站点位置等参数后，查询公交路线信息。公交线路包括公交车、地铁等多种交通方式。" \
                "目标站点由公交线路和站点位置共同确定。站点位置默认为'当前位置站点'。",
            "qa_for_realtime_mass_transit":
                "在realtime_mass_transit查询公交路线信息后，对实时公交信息进行询问，比如29路公交车什么时候到站等。",
            "call_map_native_function":
                "地图功能指令，包括但不限于播放音乐、查询当前定位、提高音量、查看路线的终点、查询限行信息、导航指令等等。", 
            "ask_for_LLM":
                "在用户query超过了地图app的服务范围时，可以调用通用语言模型来给用户一个通用的回答。"
        }
    )
    OPERATOR_DESCS_CN = '\n'.join([': '.join([k, v]) for k, v in OPERATOR_DICT_CN.items()])

    OPERATOR_OBJ_DICT_CN = dict(
        {
            "search_for_poi":
                ["名称", "地点类别", "地点特色", "位置"],
            "qa_for_poi":
                ["简介", "营业时间", "电话", "地址", "攻略"],
            "search_for_navi":
                ["终点", "起点和终点", "交通方式", "途经点", "出发时间", "到达时间"],
            "qa_for_navi":
                ["路程长度", "路程耗时", "路费"],
            "ride_hailing":
                ["终点", "起点和终点", "途经点", "出发时间", "到达时间"],
            "qa_for_ride_hailing":
                ["路程耗时", "等待时间", "预估路费"],
            "realtime_mass_transit":
                ["目标站点"],
            "qa_for_realtime_mass_transit":
                ["到站时间"],
            "call_map_native_function":
                ["查询当前定位", "导航指令", "查看终点", "查询限行信息"],
            "ask_for_LLM":
                [],
        }
    )

    CLARIFY_EXAMPLES_CN = """

下面是一些示例：

query：那旅游景点又有哪些呢
分析：用户的意图是[search_for_poi]，[位置]默认为'当前位置'，[类别]为'景点'，意图明确，因此不需要进行澄清。
需要澄清：False
任务描述：查找当前位置附近的，类别为旅游景点的POI

query6：露营研学
分析：用户的意图是[search_for_poi]，[位置]默认为'当前位置'，[特色]为'露营研学'，意图明确，因此不需要进行澄清。
需要澄清：False
任务描述：搜索当前位置附近的，特色为露营研学的POI

query：住酒店住哪里？
分析：用户的意图是[search_for_poi]，[位置]默认为'当前位置'，[类别]为'酒店'，意图明确，不需要进行澄清。
需要澄清：False
任务描述：搜索当前位置附近的，类别为酒店的POI

query：坐火车呢？
分析：用户的意图是[search_for_navi]，[交通方式]是'火车'，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里呢？

query：假如我选择高铁，有哪些车次？
分析：用户的意图是[search_for_navi]，[交通方式]是'高铁'，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里呢？

query：航班有哪些？
分析：用户的意图是[search_for_navi]，[交通方式]是'飞机'，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：北京到杭州要多少钱
分析：用户的意图是[search_for_navi]和[qa_for_navi]，[起点]是'北京'，[终点]是'杭州'，但[交通方式]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的交通方式是什么？

query：查公交
分析：用户的意图是[realtime_mass_transit]，[目标站点]不明确，站点默认为'当前位置站点'，但公交线路缺失，因此需要进行澄清。
需要澄清：True
问题：请问您想查询的公交线路是什么？

query：从软件园方舟大厦怎么过去
分析：用户的意图是[search_for_navi]，[起点]是'软件园方舟大厦'，[交通方式]默认为'驾车'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：景区里面线路推荐
分析：用户的意图是[qa_for_poi]，询问某个景区的推荐线路，qa_target='线路推荐'，但要询问的[poi_target]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要询问的景区是什么？

query：从昌平区金域华府一期出发
分析：用户的意图是[search_for_navi]，[起点]是'昌平区金域华府一期'，[交通方式]默认为'驾车'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：从北京回去推荐那种交通工具
分析：用户的意图是[search_for_navi]和[qa_for_navi]，[起点]为'北京'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：最快的高铁列车多少钱
分析：用户的意图是[search_for_navi]和[qa_for_navi]，[起点]默认为'当前位置'，[交通方式]为'高铁'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：你能调起打车服务吗
分析：用户的意图是发起打车[ride_hailing]，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：可以，请问您的终点是哪里？

query：帮我查下怎么过去
分析：用户的意图是[search_for_navi]，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：出行方案
分析：用户的意图是[search_for_navi]，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是哪里？

query：大宅门里的
分析：用户的意图不明确，因此需要进行澄清。
需要澄清：True
问题：请问您是打算搜点还是问路呢？

query：揭阳机场飞一个城市再回北京的航线
分析：用户的意图是[search_for_navi]，[起点]是'揭阳机场'，[终点]是'北京'，但[一个城市]这个[途经点]不明确，因此需要进行澄清。
需要澄清：True
问题：请问您说的一个城市是指哪里呢？

query：如何买票
分析：用户的意图是[search_for_navi]和[qa_for_navi]，[起点]默认为'当前位置'，但[终点]和[交通方式]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要去哪里？交通方式是什么呢？

query：从武汉怎么去
分析：用户的意图是[search_for_navi]，[起点]是'武汉'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您的终点是什么？

query：那里有什么玩的
分析：用户的意图是[search_for_poi]，但'那里'指代不明确，[位置]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您说的'那里'是指的哪里呢？

query：要多长时间走出来呢
分析：用户的意图是[qa_for_navi]，询问路线的耗时情况，但[目标路线]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要询问的路线是什么呢？

query：建议哪一天出去玩
分析：用户的意图不明确，可能是[search_for_navi]和[qa_for_navi]，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要去哪里玩呢？

query：带孩子去有什么注意事项
分析：用户的意图是[qa_for_poi]，[qa_target]是'带孩子去有什么注意事项'，但[poi_target]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要询问的地点是哪里呢？

query：坐飞机呢
分析：用户的意图可能是[qa_for_navi]，[交通方式]是'飞机'，[起点]默认为'当前位置'，但[终点]缺失，因此需要进行澄清。
需要澄清：True
问题：请问您要查询飞机航班吗？起点和终点是哪里呢？

query：具体到景点
分析：用户的意图不明确，因此需要进行澄清。
需要澄清：True
问题：请问您要搜索景点吗？

    """

# {assistant_role}将帮助{user_role}完成这个任务。
# {assistant_role}能够支持如下功能：
# 
# 如果{assistant_role}不能支持任务：{task}
    TASK_SPECIFY_PROMPT_CN: TextPrompt = TextPrompt(
        """这是一个任务：
'''
{task}
'''

给定功能集合如下：
<FUNCS_DESC>
{funcs_desc}
</FUNCS_DESC>

如果{assistant_role}能够支持任务：{task}，那么原任务描述是可行的，请确保生成的任务描述尽量具体，并且保证逻辑性和准确性。
请确保生成的任务描述简短，不要超过{word_limit}字，请勿添加任何与任务无关的额外内容。
输出格式为：
query：<USER_QUERY>
分析：<REASON>
需要澄清：False
任务描述：<SPECIFIED_TASK>

如果判断已知的功能集无法满足{user_role}的query，那么原任务描述不可行，需要向{user_role}进行提问，指导{user_role}对原任务描述进行澄清。
如果任务描述中有不明确的地方，请一定要向用户提出澄清问题。
对于无法完成的任务，必须向{user_role}进行提问。不能假设用户会在后续交互中会提供额外输入。
请确保生成的澄清问题与任务相关，请勿添加任何与任务无关的额外内容。
输出格式为：
query：<USER_QUERY>
分析：<REASON>
需要澄清：True
问题：<CLARIFIED_QUESTION>
{clarify_examples}
"""
    )

    TASK_SPECIFY_PROMPT_CN_V2: TextPrompt = TextPrompt(
        """这是一个任务概述：{task}。{assistant_role}将帮助{user_role}完成这个任务。
{assistant_role}能够支持如下功能：
<FUNCS_DESC>
{funcs_desc}
</FUNCS_DESC>

请确保生成的描述尽量具体，并且保证可行性、逻辑性和准确性。
如果任务描述可能使用不在<FUNCS_DESC>中的新函数，则视为任务描述不可行。
请必须按照如下两种方式输出指令：

1. 不需要澄清的任务描述：
状态：<CLEAR>
澄清问题：None
任务描述：<SPECIFIED_TASK>

2. 需要澄清的任务描述：
状态：<UNCLEAR>
澄清问题：<YOUR_QUESTION>
任务描述：None

“状态”必须是“<CLEAR>”或“<UNCLEAR>”。
"澄清问题"必须是None或<YOUR_QUESTION>。
“任务描述”必须为None或<SPECIFIED_TASK>。
当“状态”为<CLEAR>，“澄清问题”为None，“任务描述”为<SPECIFIED_TASK>，你需要给出具体的任务描述：<SPECIFIED_TASK>。
当“状态”为<UNCLEAR>，”澄清问题“为<YOUR_QUESTION>，“任务描述”为None。你需要对{user_role}给出具体的澄清问题：<YOUR_QUESTION>。

生成的问题是针对原任务描述（{task}）的澄清问题，
请确保生成的澄清问题以”TASK_UNCLEAR. “为首句。

请确保生成的任务描述简短，不要超过{word_limit}字，请勿添加任何与任务无关的额外内容。
请确保生成的澄清问题与任务相关，并且简短，不要超过{word_limit}字，请勿添加任何与任务无关的额外内容。"""
    )

    JUDGE_PROMPT_CN: TextPrompt = TextPrompt(
        """请仔细理解下列文本：
'''
{description}
'''

并判断这个陈述：
'''
{statement}
'''
为"True"还是"False"。

你的回答只能是"True"或"False"。

请给出你的判断："""
    )

    ASSISTANT_PROMPT_CN: TextPrompt = TextPrompt(
        """
请一定不要忘记你是一个{assistant_role}，而我是一个{user_role}。请务必不要颠倒角色！请避免对我发送指令！
我们需要通力合作，完成一个共同的任务："{task}"
请务必帮助我完成这个任务："{task}"
千万不要忘记我们的任务！

我应该基于我的需求和你的专业知识，指导你完成这个任务。
我一次只能给你一条指令请求。
你必须根据收到的指令请求给出准确、具体、可执行的答案。
如果由于物理的、道德的、法律的原因，或者是你自身的能力有限，导致你不能执行收到的指令请求，请务必拒绝我的指令请求，并解释拒绝执行指令请求的原因。
请务必根据收到的指令请求给出解决方案，除此之外，不要添加任何无关内容。
你不应该对我提出问题或发送指令请求。
你的解决方案应当完整、准确。
你需要解释你的解决方案。
你应该用陈述句式描述你的解决方案。
除非我说任务已完成，否则你的回答首句应该始终用如下形式：

解决方案：<YOUR_SOLUTION>

<YOUR_SOLUTION>应该具体，并包含实现方式。
请用“下一条指令。”作为你的回答内容的最后一句。"""
    )
    
    ROUTER_PROMPT_CN: TextPrompt = TextPrompt(
        """
请根据指令："{instruction}"和输入：{input}，选择合适的执行者。
可用的执行者包括：{assistant_agents}。他们具有不同的专业知识和能力：
{assistant_agent_details}

你必须基于{assistant_agents}的不同专业知识和能力，从中选择出适合执行指令（"{instruction}"）的执行者。
你只能通过以下一种方式给出指令：

原因：<YOUR_EXPLANATION>
执行者：<SELECTED_AGENT>

“原因”必须为选择“执行者”的更进一步的、具体的理由和解释。
"执行者"必须是{assistant_agents}中的一个。

你一次只能选择一个“执行者”。
现在，请开始从{assistant_agents}中做出选择。

你必须根据上面提到的形式给出选择指令。
你输出的回答只能包含“原因”和“执行者”。除此之外，你的回答不应该包含其他内容。"""
    )

    ROUTER_PROMPT_CN_BAK: TextPrompt = TextPrompt(
        """
请根据指令：{instruction}和输入：{input}，选择合适的执行者。
可用的执行者包括：{assistant_agents}。他们具有不同的专业知识和能力：
{assistant_agents_details}

你必须基于{assistant_agents}}的不同专业知识和能力，从中选择出适合执行指令（{instruction}）的执行者。
你只能通过以下一种方式给出指令：

指令：{instruction}
输入：{input}
原因：<YOUR_EXPLANATION>
执行者：<SELECTED_AGENT>

“指令”必须是原指令：{instruction}。
"输入"必须是原输入：{input}。
“原因”必须为选择“执行者”的更进一步的、具体的理由和解释。
"执行者"必须是{assistant_agents}中的一个。

你一次只能选择一个”执行者“。
现在，请开始从{assistant_agents}中做出选择。

你必须根据上面提到的形式给出选择指令。
你输出的回答只能包含“指令”，“输入”，“执行者”和“原因”。除此之外，你的回答不应该包含其他内容。
在你认为任务完成之前，请你务必要输出选择指令。
当你认为任务已经完成，你只能回复：“<TASK_DONE>”。
除非收到的指令（{instruction}）说明任务已完成，否则不要输出“<TASK_DONE>”。"""
    )

    USER_PROMPT_CN: TextPrompt = TextPrompt(
        """
你是一个{user_role}，{assistant_role}会帮助你完成任务："{task}"

你需要和{assistant_role}通力合作，完成一个共同的任务："{task}"
{assistant_role}会帮助你完成这个任务。请务必不要颠倒角色！
不要忘记这个任务！

你必须对{assistant_role}进行指导，从而完成任务："{task}"

你应该基于你的需求和{assistant_role}的专业知识和能力，指导他完成这个任务。
你只能通过以下两种方式对{assistant_role}发出执行动作的请求：

1. 包含输入的指令：
指令：<INSTRUCTION>
输入：<YOUR_INPUT>

2. 不包含输入的指令：
指令：<INSTRUCTION>
输入：None

"指令"必须是一个具体的任务或执行计划。
“输入”为描述“指令”提供更进一步的、具体的上下文信息。

你一次只能给{assistant_role}一条指令请求。
{assistant_role}必须根据收到的指令请求给出准确、具体的解决方案。
如果由于物理的、道德的、法律的原因，或者是{assistant_role}自身的能力有限，导致{assistant_role}不能执行收到的指令请求，
{assistant_role}应该拒绝你的指令请求，并解释拒绝执行你的指令请求的原因。
你应该指导{assistant_role}不要对你提出问题或发送指令请求。
现在，请开始对{assistant_role}发出指令请求。

你必须根据上面提到的两种指令形式对{assistant_role}发出指令请求。
你发送的指令请求只能包含“指令”和“输入”。除此之外，你的指令请求不应该包含其他内容。
在你认为任务完成之前，请你务必要对{assistant_role}发送指令请求。
当你认为任务已经完成，你只能回复{assistant_role}：“<TASK_DONE>”。
除非{assistant_role}的回复解决了任务："{task}"，否则不要对{assistant_role}回复“<TASK_DONE>”。"""
    )

    USER_PROMPT_PLAN_CN: TextPrompt = TextPrompt(
        """
你是一个{user_role}，{assistant_role}会帮助你完成任务："{task}"

你需要和{assistant_role}通力合作，完成一个共同的任务："{task}"
{assistant_role}会帮助你完成这个任务。请务必不要颠倒角色！
不要忘记这个任务！

请务必要根据{assistant_agents}中不同成员的专业能力来判断哪个更适合作为{assistant_role}的执行者来完成任务：{task}

{assistant_agents}具有不同的专业知识和能力，因此你需要根据他们的不同特点做出选择。
{assistant_agents}中不同成员的专业能力描述如下：
{assistant_agent_details}

你必须对{assistant_role}进行指导，并选择合适的执行者，从而完成任务："{task}"
请给出选择执行者的理由。

你应该基于你的需求和{assistant_role}的专业知识和能力，指导他完成这个任务。
你只能通过以下两种方式对{assistant_role}发出执行动作的请求：

1. 包含输入的指令：
指令：<INSTRUCTION>
输入：<YOUR_INPUT>
原因：<SELECT_AGENT_REASON>
执行者：<SELECTED_AGENT>

2. 不包含输入的指令：
指令：<INSTRUCTION>
输入：None
原因：<SELECT_AGENT_REASON>
执行者：<SELECTED_AGENT>

"指令"必须是一个具体的任务或执行计划。
“输入”为描述“指令”提供更进一步的、具体的上下文信息。
“原因”是选择执行者的理由。
“执行者”必须是{assistant_agents}中的一个成员。

你一次只能给{assistant_role}一条指令请求。
{assistant_role}必须根据收到的指令请求给出准确、具体的解决方案。
如果由于物理的、道德的、法律的原因，或者是{assistant_role}自身的能力有限，导致{assistant_role}不能执行收到的指令请求，
{assistant_role}应该拒绝你的指令请求，并解释拒绝执行你的指令请求的原因。
你应该指导{assistant_role}不要对你提出问题或发送指令请求。
现在，请开始对{assistant_role}发出指令请求。

你必须根据上面提到的两种指令形式对{assistant_role}发出指令请求。
你发送的指令请求只能包含“指令”，“执行者”，”选择执行者的理由“和“输入”。
除此之外，你的指令请求不应该包含其他内容。
在你认为任务完成之前，请你务必要对{assistant_role}发送指令请求。
当你认为任务已经完成，你只能回复{assistant_role}：“<TASK_DONE>”。
除非{assistant_role}的回复解决了任务："{task}"，否则不要对{assistant_role}回复“<TASK_DONE>”。"""
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update({
            "task_specify_prompt": self.TASK_SPECIFY_PROMPT_CN.format(
                                                            funcs_desc=self.OPERATOR_DESCS_CN, 
                                                            clarify_examples=self.CLARIFY_EXAMPLES_CN),
            "judge_prompt": self.JUDGE_PROMPT_CN,
            "router_proompt": self.ROUTER_PROMPT_CN,
            RoleType.ASSISTANT: self.ASSISTANT_PROMPT_CN,
            RoleType.USER: self.USER_PROMPT_CN,
            RoleType.USER_PLAN: self.USER_PROMPT_PLAN_CN
        })
