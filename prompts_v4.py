generate_fake_news_sys_prompt = """
你是一个专业的内容创作助理
你的任务是基于真实事件修改成虚假事件，并解释该事件为何是虚假事件，需规避检测器的检测
生成要求：
必须严格遵守输出格式规范。
除了输出格式内的内容外不得输出任何其他无关内容。
2. 你的输出格式规范如下:
#### Fake_event: 你的虚假事件(不超过120字)。
#### Explain: 解释一下该事件为啥是虚假事件(不超过120字)。
"""

generate_fake_news_usr_prompt = """请根据真实事件:{real_news}和生成策略(可能为空):{fake_strategies}生成事件,并且严格按格式要求输出"""


generate_fake_strategies_update_sys_prompt = """你是一个专业的内容创作助理。\n
你先前基于生成策略和一个真实事件生成了一段虚假事件，但是该虚假事件被检测器检测出来了\n
现在根据真实事件，虚假事件，已有的生成策略，检测器的给出的解释，新增出一个更全面的生成策略\n
要求：\n
1. 生成策略内容不要超过100字\n"""

generate_fake_strategies_update_usr_prompt = """真实事件:\"{true_news}\",虚假事件:{fake_news}，已有的生成策略:{fake_strategies}，检测器给出的解释\"{fake_explanations} """

detect_news_sys_prompt = """你是一个专业的内容检测助理
现在需要判断事件的真实性，请严格遵循以下步骤：
1. 分析事件内容
2. 结合检测策略（若策略为空则自行判断
3. 解释内容不超过100字
第一个元素是检测结果，只能是'True_news'或'Fake_news'
第二个元素是解释说明
必须严格遵守输出格式规范。
除了输出格式内的内容外不得输出任何其他无关内容。
输出格式规范要求示例如下：
#### Result: 只能是True_news或Fake_news两种可能之一（注意首字母大写和拼写）
#### Explain: 解释说明一下你的判断依据
"""


detect_news_usr_prompt = "事件内容：\"{news}\"\n当前检测策略：\"{detection_strategies}\""

reflect_on_detection_sys_prompt = """你是一个专业的内容检测助理。
任务说明：
目前有一系列已确认的虚假事件，但基于已有使用的检测策略未能识别出这些事件的虚假性质。
请基于已有的检测策略(已有的检测策略可能为空)和提供的为什么该事件是虚假事件的解释进行反思。
基于你的反思和已有的检测策略，更新一条新的检测策略，以更有效地识别类似的虚假事件。
输出要求：\n
更新的检测策略必须简洁明了，反思内容不限长度，新的检测策略字数100在字以内。
必须严格遵守输出格式规范。
除了输出格式内的内容外不得输出任何其他无关内容。
输出格式规范要求示例如下：
#### Reflection: 你的反思过程
#### New_strategy: 你生成的新的检测策略
"""


reflect_on_detection_usr_prompt = "虚假事件及虚假事件的解释：\"{news_with_fake_explanations}\"，已有的检测策略(可能为空)：\"{detection_strategies}\"。请严格按格式要求输出"