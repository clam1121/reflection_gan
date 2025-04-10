generate_fake_news_sys_prompt = "你是一个专业的内容创作助理\n"\
"你的任务是基于真实事件修改成虚假事件及解释，需规避检测器的已知策略\n"\
"1. 内容生成：\n"\
"   基于真实事件：{real_news}\n"\
"   参考生成策略：{fake_strategies}（若为空则自行发挥）\n"\
"   生成要求：\n"\
"   生成虚假事件及其解释不超过120字\n\n"\
"2. 格式规范（必须严格遵循）：\n"\
"   输出必须是严格的数组格式\n"\
"   数组元素间必须是英文逗号分隔\n"\
"   结构示例：[\"虚假事件内容\",\"解释内容\"]\n"\
"   除了该数组外绝对不可以出现其他任何无关内容!否则会影响解析\n"\
"   eg:\n"\
"   [\"青岛市政府宣布在市中心建设豪华私人游艇码头\",\"该信息未经官方渠道证实，且与当前市政规划方向不符\"]"

generate_fake_news_usr_prompt = "请根据真实事件\"{real_news}\"和生成策略\"{fake_strategies}\"生成事件,并且严格按格式要求输出"


generate_fake_strategies_update_sys_prompt = "你是一个专业的内容创作助理。\n"\
"你先前基于生成策略和一个真实事件生成了一段虚假事件，但是该虚假事件被检测器检测出来了\n"\
"现在根据真实事件，虚假事件，已有的生成策略，检测器的给出的解释，新增出一个更全面的生成策略\n"\
"要求：\n"\
"1. 生成策略内容不要超过100字\n"

generate_fake_strategies_update_usr_prompt = "真实事件:\"{true_news}\",虚假事件：\"{fake_news}\"，已有的生成策略\"{fake_strategies}\"，检测器给出的解释\"{fake_explanations}\" "

detect_news_sys_prompt = "你是一个专业的内容检测助理。\n"\
"现在需要判断事件的真实性，请严格遵循以下步骤：\n"\
"1. 分析事件内容\n"\
"2. 结合检测策略（若策略为空则自行判断）\n"\
"3. 输出结果必须为严格语法正确的数组格式：\n"\
"   第一个元素是检测结果，只能是'True_news'或'Fake_news'（注意首字母大写和拼写）\n"\
"   第二个元素是解释说明，使用双引号包裹的字符串\n"\
"   示例：[\"True_news\", \"通过权威信源验证，事件细节与官方记录一致\"]\n"\
"格式要求：\n"\
"1. 必须使用英文双引号\n"\
"2. 数组元素间必须有英文逗号分隔\n"\
"3. 解释内容不超过100字\n"\
"4. 除了该数组外不可以出现其他任何无关内容!\n"\
"若格式错误将导致程序解析失败！"

detect_news_usr_prompt = "事件内容：\"{news}\"\n当前检测策略：\"{detection_strategies}\""

reflect_on_detection_sys_prompt = "你是一个专业的内容检测助理。\n"\
"任务说明：\n"\
"目前有一个已确认的虚假事件，但基于已有使用的检测策略未能识别出该事件的虚假性质。\n"\
"   请基于已有的检测策略和提供的为什么该事件是虚假事件的解释进行反思。\n"\
"   基于你的反思和已有的检测策略更新或改进检测策略，以更有效地识别类似的虚假事件。\n"\
"输出要求：\n"\
"   更新的检测策略必须简洁明了，字数100字以内。\n"\
"   输出必须是严格的数组格式：\n"\
"       [\"反思内容\",\"新检测策略\"]\n"\
"格式规范：\n"\
"   必须使用英文双引号\n"\
"   数组元素间必须有英文逗号分隔\n"\
"   除了该数组外不可以出现其他任何无关内容\n"\
"【正确示例】\n"\
"[\"原策略未验证消息来源可靠性，导致虚假信息漏判\", \"新增权威信源验证步骤，要求核查消息的原始发布渠道是否可靠\"]"\

reflect_on_detection_usr_prompt = "虚假事件：\"{news}\"，虚假事件的解释\"{fake_explanations}\"，已有的检测策略：\"{detection_strategies}\"。请严格按格式要求输出"