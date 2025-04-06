import json

def extract_outputs_from_json(input_file_path, output_file_path):
    # 读取JSON文件
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取output字段
    outputs = [item.get('output', '') for item in data]

    # 将输出写入新的JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(outputs, file, ensure_ascii=False, indent=4)

    return outputs

# 使用示例
input_file_path = 'news.json'  # 输入文件路径
output_file_path = 'output_news.json'  # 输出文件路径
outputs = extract_outputs_from_json(input_file_path, output_file_path)
