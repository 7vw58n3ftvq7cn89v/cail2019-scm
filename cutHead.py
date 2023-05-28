import json
import re

def remove_description(text):
    pattern = r'\n\n'
    text_parts = re.split(pattern, text)
    if len(text_parts) == 2:
        return text_parts[1]
    else:
        return text

def process_dataset(input_file_path, output_file_path):
    # 读取输入文件
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    # 处理训练集中的每一个案例
    for case in data:
        case['A'] = remove_description(case['A'])
        case['B'] = remove_description(case['B'])
        case['C'] = remove_description(case['C'])

    # 将处理后的数据保存到输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)

input_file_path = "scmDataset/train.json"
output_file_path = "scmDatasetCut/train.json"

process_dataset(input_file_path, output_file_path)

input_file_path = "scmDataset/valid.json"
output_file_path = "scmDatasetCut/valid.json"

process_dataset(input_file_path, output_file_path)

input_file_path = "scmDataset/test.json"
output_file_path = "scmDatasetCut/test.json"

process_dataset(input_file_path, output_file_path)
