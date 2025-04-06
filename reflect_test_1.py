# 导入必要的库和模块
import requests  # 用于从网站获取新闻数据（示例，实际可能需要更合适的库）
from openai import OpenAI
# from langchain.llms import OpenAI  # 假设使用OpenAI的LLM，实际需根据选择的LLM进行调整
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
import ast
import random
import json
from tqdm import tqdm
import os
import logging
from logging.handlers import RotatingFileHandler

def initialize_logger(log_dir='/data/dell/cxz_file/reflection/llm_output_log', log_file='output.log', max_bytes=10**6, backup_count=5):
    """
    Initializes a logger to record logs into a specified directory.

    :param log_dir: Directory where the log files will be stored.
    :param log_file: Name of the log file.
    :param max_bytes: Maximum size (in bytes) before log rotation occurs.
    :param backup_count: Number of rotated log files to keep.
    :return: Configured logger object.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logger
    logger = logging.getLogger('LLMLogger')
    logger.setLevel(logging.DEBUG)

    # Create a rotating file handler
    log_path = os.path.join(log_dir, log_file)
    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)

    # Define the format for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

# Example usage:
logger = initialize_logger()
# Now you can use this logger to record messages
logger.info('This is an info message.')
logger.error('This is an error message.')

def prompt_to_string(prompt):
    """
    Convert a structured prompt with roles and content to a single string.

    :param prompt: A list of tuples where each tuple contains a role and a message.
                   For example: [("system", "content"), ("user", "content")]
    :return: A single string with each role-message pair on a new line.
    """
    output = []
    for role, message in prompt:
        output.append(f"{role}: {message}")

    return "\n\n".join(output)

llm = Ollama(model = "qwen2.5:14b")

def extract_array_from_response(response):
    try:
        res = ast.literal_eval(response)
        if(len(res) >= 2):
            return res
        else:
            return ["None", "None"]
    except (ValueError, SyntaxError):
        print("无法解析响应为数组:   "  + response)
        return ["None", "None"]

def get_response_kw(user_prompt):
    client = OpenAI(
        api_key='sk-x9DvPqXKGU2KrEkoCe173a9f89324c938d489834196eB78a', # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url='https://aium.cc/v1/',  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        )
    # return completion
    return completion.choices[0].message.content

# 1. 数据收集与预处理
def collect_real_news():
    # 从Sina News网站收集真实新闻数据（示例，实际需根据网站结构和数据获取方式调整）
    news_data = requests.get("https://news.sina.com.cn/").content
    # 进行数据清洗、标记等预处理操作
    preprocessed_news = preprocess_news(news_data)
    return preprocessed_news

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"错误：'{file_path}' 不是有效的JSON文件")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None


def write_json_to_file(data, filename='data.json'):

    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"数据已成功写入 {filename} 文件")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def split_data(real_news, fake_news, random_seed=42):
    """
    将真假新闻数据集随机打乱后划分为训练集、验证集和测试集

    Args:
        real_news: 真新闻列表
        fake_news: 假新闻列表
        random_seed: 随机种子，默认为42Returns:
        train_real: 真新闻训练集
        val_real: 真新闻验证集
        test_real: 真新闻测试集
        train_fake: 假新闻训练集
        val_fake: 假新闻验证集
        test_fake: 假新闻测试集
    """
    # 设置随机种子保证可重复性
    random.seed(random_seed)

    # 复制列表���打乱
    real_shuffled = real_news.copy()
    fake_shuffled = fake_news.copy()
    random.shuffle(real_shuffled)
    random.shuffle(fake_shuffled)

    # 计算划分点
    real_len = len(real_shuffled)
    fake_len = len(fake_shuffled)

    # 真新闻划分
    real_train_end = int(real_len * 0.7)
    real_val_end = int(real_len * 0.8)
    train_real = real_shuffled[:real_train_end]
    val_real = real_shuffled[real_train_end:real_val_end]
    test_real = real_shuffled[real_val_end:]

    # 假新闻划分
    fake_train_end = int(fake_len * 0.7)
    fake_val_end = int(fake_len * 0.8)

    train_fake = fake_shuffled[:fake_train_end]
    val_fake = fake_shuffled[fake_train_end:fake_val_end]
    test_fake = fake_shuffled[fake_val_end:]

    return train_real, val_real, test_real, train_fake, val_fake, test_fake

def split_datasets(real_news, fake_news):
    # 按照论文中的方式划分训练、验证和测试集
    train_real, val_real, test_real, train_fake, val_fake, test_fake = split_data(real_news, fake_news)
    return train_real, val_real, test_real, train_fake, val_fake, test_fake

# 2. 模型构建
# 构建生成器
def generate_fake_news(real_news, fake_strategies):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一个专业的内容创作助理。\n"\
                "请基于给出的真实事件"\
                "结合给出的策略（如果策略为空可自行发挥）\n"\
                "生成一个虚假事件并解释为什么该事件是虚假事件。\n"\
                "要求：\n"\
                "1. 虚假事件和解释各自字数不超过120字，并且都是字符串形式\n"\
                "2. 请严格按照数组形式输出,格式如下[\"虚假事件\", \"解释为什么该事件是虚假事件\"]\n"\
                "3. 除了该数组以外，不要输出任何无关内容。\n"\
                ),
            ("user","请根据真实事件：{real_news}，已有的生成策略{fake_strategies}完成任务 ")
        ]
    )
    input_prompt = prompt_to_string(prompt)
    logger.info("generate_fake_news\n" + input_prompt + "\n")
    # prompt_1 = (
    #             "你是一个专业的内容创作助理。\n"
    #             f"请基于以下真实事件：{real_news}\n"
    #             f"使用以下策略（如果策略为空可自行发挥）：{fake_strategies}\n"
    #             "生成一个虚假事件及其生成策略。\n\n"
    #             "要求：\n"
    #             "1. 虚假事件和生成策略各自字数不超过120字\n"
    #             "2. 请以数组形式输出：['虚假事件', '生成策略']\n"
    #         )
    chain_1 = prompt|llm

    response = chain_1.invoke({'real_news':real_news,'fake_strategies':fake_strategies})
    res = extract_array_from_response(response)
    logger.info("generate_fake_news_output\n" + str(res[0]) + "\n" + str(res[1]))
    return res

# 构建生成器策略更新
def generate_fake_strategies_update(true_news,fake_news, fake_strategies, fake_explanations):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一个专业的内容创作助理。\n"\
                "你先前基于生成策略和一个真实事件生成了一段虚假事件，但是该虚假事件被检测器检测出来了\n"\
                "现在根据真实事件，虚假事件，已有的生成策略，检测器的给出的解释，更新出一个更全面的生成策略\n"\
                "要求：\n"
                "1. 生成策略内容不要过长\n"
                "2. 生成策略必须严格按照以下格式：1,xxxxxxxxxx. 2,xxxxxxxxx 3,xxxxxxxxx .......... "
                ),
            ("user","真实事件:{true_news},虚假事件：{fake_news}，已有的生成策略{fake_strategies}，检测器给出的解释{fake_explanations} ")
        ]
    )
    input_prompt = prompt_to_string(prompt)
    logger.info("generate_fake_strategies_update\n" + input_prompt + "\n")

    chain_1 = prompt|llm
    res = chain_1.invoke({'true_news':true_news,'fake_news':fake_news,'fake_explanations':fake_explanations,'fake_strategies':fake_strategies})
    logger.info("generate_fake_strategies_update_output\n" + res)
    return res

def detect_news(news, detection_strategies):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一个专业的内容检测助理。\n"\
                "现在给出一个事件"
                "请基于已有的检测策略（如果策略为空可自行发挥）判断事件的真实性\n"\
                "如果鉴别出来是假事件则返回'Correct',未鉴别出来则返回'Incorrect'\n"\
                "并且给出你的解释。\n"
                "要求：\n"
                "1. 解释字数不超过300字\n"
                "2. 请严格按照数组形式输出，数组的第一个元素是你的鉴别结果，鉴别结果只能为'Correct'和'Incorrect'两种可能；第二个元素是你给出的解释.格式如下[\"xxxx\", \"xxxxx\"]\n"\
                "3. 除了该数组以外，不要输出任何无关内容。\n"\
                ),
            ("user","事件：{news}，已有的检测策略：{detection_strategies} ")
        ]
    )

    input_prompt = prompt_to_string(prompt)
    logger.info("detect_news\n" + input_prompt + "\n")
    chain_1 = prompt|llm
    response = chain_1.invoke({'news':news,'detection_strategies':detection_strategies})
    res = extract_array_from_response(response)

    logger.info("detect_news\n" + res[0] + "\n" + res[1])
    return [str(res[0]), str(res[1])]

# 构建反思器
def reflect_on_detection(news, detection_strategies,fake_explanations):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一个专业的内容检测助理。\n"\
                "任务说明：\n"\
                "目前有一个已确认的虚假事件，但基于已有使用的检测策略未能识别出该事件的虚假性质。\n"\
                "- 请基于已有的检测策略和提供的为什么该事件是虚假事件的解释进行反思。\n"\
                "- 基于你的反思和已有的检测策略更新或改进检测策略，以更有效地识别类似的虚假事件。\n"\
                "输出要求：\n"\
                "- 更新的检测策略必须简洁明了，字数不限。\n"\
                ),
            ("user","虚假事件：{news}，虚假事件的解释{fake_explanations}，已有的检测策略：{detection_strategies}完成任务 ")
        ]
    )

    input_prompt = prompt_to_string(prompt)
    logger.info("reflect_on_detection\n" + input_prompt + "\n")

    chain_1 = prompt|llm
    res = chain_1.invoke({'news':news,'detection_strategies':detection_strategies,'fake_explanations':fake_explanations})
    logger.info("reflect_on_detection_output\n" + res)
    return res


def build_llm_gan():
    # 构建生成器
    # def generate_fake_news(real_news, fake_strategies):
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system","你是一个专业的内容创作助理。\n"\
    #                 "请基于给出的真实事件\n"\
    #                 "使用给出的策略（如果策略为空可自行发挥）\n"\
    #                 "生成一个虚假事件及其生成策略。\n\n"
    #                 "要求：\n"
    #                 "1. 虚假事件和生成策略各自字数不超过120字\n"
    #                 "2. 请以数组形式输出：['虚假事件', '生成策略']\n"
    #                 ),
    #             ("user",f"请根据真实事件：{real_news}，已有的生成策略{fake_strategies}完成任务 ")
    #         ]
    #     )
    #     chain_1 = prompt_1|llm
    #     response = chain_1.invoke()
    #     res = extract_array_from_response(response)
    #     return res

        # messages = [
        #     {"role": "system", "content": "你是一个有用的助理."},
        #     {"role": "user", "content": f"在基于以下事件的基础上生成虚假事件: {real_news} 可以基于以下策略生成（如果策略为空可以自己随意生成）s: {fake_strategies}。字数限制不超过80字。"}
        # ]
        # client = OpenAI(
        # api_key='sk-x9DvPqXKGU2KrEkoCe173a9f89324c938d489834196eB78a', # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        # base_url='https://aium.cc/v1/',  # 填写DashScope服务的base_url
        # )
        # completion = client.chat.completions.create(
        # model='gpt-4o-mini',
        # messages= messages
        # )
        # return completion.choices[0].message.content


    # 构建检测器
    def detect_news(news, detection_strategies):
        messages = [
            {"role": "system", "content": "你是一个有用的助理."},
            {"role": "user", "content": f"以下事件可能是真的吗? {news} 可以参考以下检测策略判断是否为真的: {detection_strategies},若为真，直接返回'Correct',若为假，返回'Incorrect',不要回复任何无关内容"}
        ]
        client = OpenAI(
        api_key='sk-x9DvPqXKGU2KrEkoCe173a9f89324c938d489834196eB78a', # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url='https://aium.cc/v1/',  # 填写DashScope服务的base_url
        )
        completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages= messages
        )
        return completion.choices[0].message.content

    # 构建反射器
    def reflect_on_detection(news, detection_explanations):
        messages = [
            {"role": "system", "content": "你是一个有用的助理"},
            {"role": "user", "content": f"反思此事件的检测结果: {news} 通过以下解释: {detection_explanations}"}
        ]
        client = OpenAI(
        api_key='sk-x9DvPqXKGU2KrEkoCe173a9f89324c938d489834196eB78a', # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url='https://aium.cc/v1/',  # 填写DashScope服务的base_url
        )
        completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages= messages
        )
        return completion.choices[0].message.content
    
    # 更新检测器策略
    def detect_strategies(news, strategy):
        messages = [
            {"role": "system", "content": "你是一个有用的助理"},
            {"role": "user", "content": f"以下事件可能是真的吗? {news} 可以参考以下检测策略判断是否为真的: {detection_explanations},若为真，直接返回'Correct',若为假，返回'Incorrect',不要回复任何无关内容"}
        ]

    return generate_fake_news, detect_news, reflect_on_detection

# 3. 对抗式提示（Inter - adversary Prompting）
def inter_adversary_prompting(generator_chain, detector_chain, real_news, fake_strategies, detection_strategies):
    # 生成器生成假新闻
    fake_news, fake_explanations = generator_chain.run(real_news=real_news, fake_strategies=fake_strategies)
    print(fake_news)
    # 检测器检测假新闻
    detection_result, detection_explanations = detector_chain.run(news=fake_news, detection_strategies=detection_strategies)

    # 根据检测结果更新生成器和检测器策略
    if detection_result == "Correct":
        # 更新生成器策略
        new_fake_strategies = generator_chain.run(real_news=real_news, fake_explanations=detection_explanations)
        return new_fake_strategies, detection_strategies
    else:
        # 更新检测器策略
        new_detection_strategies = detector_chain.run(news=fake_news, fake_explanations=fake_explanations)
        return fake_strategies, new_detection_strategies

# 4. 自我反思提示（Self - reflection Prompting）
def self_reflection_prompting(detector_chain, reflector_chain, news, detection_strategies):
    # 检测器检测新闻
    detection_result, detection_explanations = detector_chain.run(news=news, detection_strategies=detection_strategies)

    # 如果检测错误，进行自我反思
    if detection_result == "Incorrect":
        reflection = reflector_chain.run(news=news, detection_explanations=detection_explanations)
        # 更新检测器策略
        new_detection_strategies = detector_chain.run(news=news, reflection=reflection)
        return new_detection_strategies
    else:
        return detection_strategies

# 5. 模型训练与评估
def train_llm_gan( train_real, train_fake, val_real, val_fake, Correct_distance, Incorrect_distance, num_epochs):

    # 初始化生成器和检测器策略
    fake_strategies = None
    detection_strategies = None
    fake_strategies_list = []
    detection_strategies_list = []
    accuracy_list = list()

    # 训练循环
    for epoch in range(num_epochs):
        fake_strategies_list.append(f"第{epoch+1}轮")
        detection_strategies_list.append(f"第{epoch+1}轮")
        Correct_step = 0
        Incorrect_step = 0
        for real_news in tqdm(train_real, desc=f"Training Epoch {epoch+1}", unit="sample"):
            # 使用生成器生成假新闻
            fake = generate_fake_news(real_news, fake_strategies)
            fake_news = fake[0]
            fake_explanations = fake[1]
            # 使用检测器检测假新闻
            detection = detect_news(fake_news, detection_strategies)
            detection_result = detection[0]
            detection_explanation = detection[1]
            # 根据检测结果更新生成器和检测器策略
            if detection_result == "Correct":
                # 更新生成器策略
                # if fake_strategies != None:
                Correct_step += 1
                if Correct_step%Correct_distance == 0:
                    fake_strategies = generate_fake_strategies_update(real_news, fake_news, fake_strategies, detection_explanation)
                    fake_strategies_list.append(fake_strategies)
                    # print(f"fake_strategies:{fake_strategies}")
            if detection_result == "Incorrect":
                # 更新检测器策略
                Incorrect_step += 1
                if Incorrect_step%Incorrect_distance == 0:
                    detection_strategies = reflect_on_detection(fake_news, detection_strategies,fake_explanations)
                    detection_strategies_list.append(detection_strategies)
                    # print(f"detection_strategies:{detection_strategies}")
        Correct_accuracy = Correct_step/len(train_real)
        Incorrect_accuracy = Incorrect_step/len(train_real)
        accuracy_list.append([Correct_accuracy, Incorrect_accuracy])
        detection_strategies_list.append(f"*****************准确率为{Correct_accuracy}*****************")
        # for news in train_fake:
        #     # 自我反思提示
        #     detection_result = detector_chain(news, detection_strategies)
        #     if detection_result == "Incorrect":
        #         reflection = reflector_chain(news, detection_result)
        #         # 更新检测器策略
        #         detection_strategies = detector_chain(news, reflection)

        # # 在验证集上评估模型（示例，实际评估指标计算需更详细实现）
        # val_metrics = evaluate_model(detector_chain, val_real, val_fake)
        # print(f"Epoch {epoch}: Validation Metrics - {val_metrics}")
    return fake_strategies_list,detection_strategies_list,accuracy_list
def evaluate_model(detector_chain, real_news, fake_news):
    # 计算准确率、F1分数等评估指标（示例，实际需完整实现评估逻辑）
    correct_predictions = 0
    total_predictions = 0
    for news in real_news + fake_news:
        detection_result, _ = detector_chain.run(news=news, detection_strategies=detection_strategies)
        if (news in real_news and detection_result == "Real") or (news in fake_news and detection_result == "Fake"):
            correct_predictions += 1
        total_predictions += 1
    accuracy = correct_predictions / total_predictions
    return {"Accuracy": accuracy}

if __name__ == "__main__":
    # 收集真��新闻数据
    real_news = load_json_data("data/output_news_100.json")
    fake_news = load_json_data("fake_news_train.json")
    # 划分数据集
    train_real, val_real, test_real, train_fake, val_fake, test_fake = split_datasets(real_news, fake_news)
    print(len(train_real))
    print(len(train_fake))
    print(len(val_real))
    print(len(val_fake))
    print(len(test_real))
    print(len(test_fake))
    Correct_distance = 15
    Incorrect_distance = 1
    num_epochs = 20
    res_1, res_2, res_3 = train_llm_gan(train_real, train_fake, val_real, val_fake, Correct_distance, Incorrect_distance, num_epochs)
    write_json_to_file(res_1, f"fake_strategies_Correct_distance{Correct_distance}_Incorrect_distance{Incorrect_distance}")
    write_json_to_file(res_2, f"detection_strategies_Correct_distance{Correct_distance}_Incorrect_distance{Incorrect_distance}")
    write_json_to_file(res_3, f"accuracy_list_Correct_distance{Correct_distance}_Incorrect_distance{Incorrect_distance}")

    # # 构建LLM - GAN模型
    # generator_chain, detector_chain, reflector_chain = build_llm_gan()

    # # 训练LLM - GAN模型
    # train_llm_gan(generator_chain, detector_chain, reflector_chain, train_real, train_fake, val_real, val_fake)

    # # 在测试集上评估模型（示例，实际需完整实现评估逻辑）
    # test_metrics = evaluate_model(detector_chain, test_real, test_fake)
    # print(f"Test Metrics - {test_metrics}")


# if __name__ == "__main__":
#     # print(get_response_kw("你是谁？"))
#     llm = ChatOpenAI(
#         api_key='sk-x9DvPqXKGU2KrEkoCe173a9f89324c938d489834196eB78a', # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#         base_url='https://aium.cc',  # 填写DashScope服务的base_url
#         model='gpt-4o-mini',
#     )
#     generator_prompt_template = PromptTemplate(
#         input_variables=["real_news", "fake_strategies"],
#         template="Generate fake news based on this real news: {real_news} and these fake strategies: {fake_strategies}"
#     )
#     generator_chain = LLMChain(llm=llm, prompt=generator_prompt_template)
#     real_news = "2024年世界经理人峰会暨“世界品牌500强”发布会日前在香港举行，中国（包含港澳台）50个品牌上榜稳居全球第三"
#     fake_strategies = "暂无"
#     fake_news = generator_chain.run(real_news=real_news, fake_strategies=fake_strategies)