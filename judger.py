from sentence_transformers import SentenceTransformer, InputExample, losses,models
from myModel import ScmSentenceTransformer
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers.evaluation import TripletEvaluator

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data:
        a = item['A']
        label = item['label']
        if label == 'B':
            b = item['B']
            c = item['C']
        else:
            c = item['B']
            b = item['C']
        examples.append(InputExample(texts=[a, b, c]))
    

    return examples

test_path = "scmDatasetCut/test.json"
model_path = "Trained/model_cut_maxweight_lstm_cnn_0.7324"  # 请确保此路径指向您保存的训练好的模型
"sbert/Trained/model_cut_maxweight_lstm_cnn_0.7324"
# 在这里指定GPU
device = torch.device("cuda:0")

# 加载模型时将其分配给所需的GPU
model = ScmSentenceTransformer(model_path)


test_examples = load_dataset(test_path)  # 使用与训练和验证数据集相同的函数加载测试数据集

correct_predictions = 0
total_predictions = len(test_examples)


for example in tqdm(test_examples, desc="Evaluating"):
    anchor, positive, negative = example.texts

    # 单GPU
    anchor_embedding = model.encode(anchor)
    positive_embedding = model.encode(positive)
    negative_embedding = model.encode(negative)


    # 在计算距离之前，将张量移动到GPU上
    anchor_embedding = torch.from_numpy(anchor_embedding).to(device)
    positive_embedding = torch.from_numpy(positive_embedding).to(device)
    negative_embedding = torch.from_numpy(negative_embedding).to(device)

    # anchor_embedding = torch.from_numpy(anchor_embedding)
    # positive_embedding = torch.from_numpy(positive_embedding)
    # negative_embedding = torch.from_numpy(negative_embedding)

    positive_distance = torch.norm(anchor_embedding - positive_embedding)
    negative_distance = torch.norm(anchor_embedding - negative_embedding)

    if positive_distance <= negative_distance:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Accuracy on test set: {accuracy * 100:.4f}%")


