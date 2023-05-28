import random
import json
import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses
from sentence_transformers.losses import TripletDistanceMetric
from sentence_transformers.evaluation import TripletEvaluator
from transformers import AutoConfig
from myModel import ScmSentenceTransformer


# os.environ['CUDA_VISIBLE_DEVICES'] = "4"


def augment(data):
    augmented_data = []

    for item in data:
        a, b, c, label = item['A'], item['B'], item['C'], item['label']

        # 原始数据
        augmented_data.append(item)

        # 反对称增广
        # augmented_data.append({"A": a, "B": c, "C": b, "label": "C" if label == "B" else "B"})

        # 自反性增广
        augmented_data.append({"A": c, "B": c, "C": a, "label": "B"})

        # 自反性+反对称增广
        # augmented_data.append({"A": c, "B": a, "C": c, "label": "C"})

        # 启发式增广
        if label == "B":
            augmented_data.append({"A": b, "B": a, "C": c, "label": "B"})
        else:
            augmented_data.append({"A": c, "B": b, "C": a, "label": "C"})

        # 启发式+反对称增广
        # if label == "B":
        #     augmented_data.append({"A": b, "B": c, "C": a, "label": "C"})
        # else:
        #     augmented_data.append({"A": c, "B": a, "C": b, "label": "B"})

    # 删除重复项
    augmented_data = [dict(t) for t in {tuple(d.items()) for d in augmented_data}]

    # 随机化顺序
    random.shuffle(augmented_data)

    return augmented_data



def load_dataset(file_path,aug=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if aug:
        data = augment(data)

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




def load_local_model(model_path):
    # 获取预训练模型的配置信息
    config = AutoConfig.from_pretrained(model_path)

    # 使用预训练模型的配置创建CustomSentenceTransformer对象
    local_model = ScmSentenceTransformer(model_path)

    # 将模型转移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)

    return local_model

# 设置日志记录器
output_path = "model"
best_path =output_path + "/model_best"
logging.basicConfig(filename=output_path + "/training.log", level=logging.INFO, format="%(asctime)s - %(message)s")


# 1. 准备训练和验证数据
train_path = "scmDatasetCut/train.json"
test_path = "scmDatasetCut/test.json"
valid_path = "scmDatasetCut/valid.json"

# train_path = "scmDataset/train.json"
# test_path = "scmDataset/test.json"
# valid_path = "scmDataset/valid.json"

train_examples = load_dataset(train_path,aug=False)
val_triplets = load_dataset(valid_path)
test_triplets = load_dataset(test_path)

# 2. 加载预训练模型 
# 使用您的本地预训练模型或提供一个受支持的预训练模型名称
model_path = "ms"  
# model_path = "model_cut_maxweight_cnn_lstm"

model = load_local_model(model_path)


# 4. 创建数据加载器
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

# 5. 定义损失函数
# 尝试不同边界值 2.5 5 7.5
margin = 1.5 
# train_loss = losses.TripletLoss(model, margin)
# distance_metric : COSINE EUCLIDEAN MANHATTAN 
train_loss = losses.TripletLoss(model,distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin= margin)

# 6. 定义评估器
anchors = [example.texts[0] for example in val_triplets]
positives = [example.texts[1] for example in val_triplets]
negatives = [example.texts[2] for example in val_triplets]
evaluator = TripletEvaluator(anchors, positives, negatives)

anchors = [example.texts[0] for example in test_triplets]
positives = [example.texts[1] for example in test_triplets]
negatives = [example.texts[2] for example in test_triplets]
evaluator_test = TripletEvaluator(anchors, positives, negatives)

# 7. 定义回调函数
def on_epoch_end(epoch):
    score = evaluator(model)
    print(f"Epoch: {epoch}, Evaluation Score: {score:.4f}")


# 8. 训练模型
num_epochs = 16
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10%的训练数据用于warmup
highest_score = 73.00

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader)/2 + 1,
        warmup_steps=warmup_steps,
        output_path=output_path,
    )
    
    
    print("Evaluating...")
    val_scores = evaluator(model)
    test_scores = evaluator_test(model)
    print(f"valid Score: {val_scores:.4f} test score: {test_scores:.4f}")

    if test_scores > highest_score :
        model.save(best_path)
        highest_score = test_scores
        print("New best model saved.")
        logging.info(f"New best model: {test_scores:.4f}\n")


    # 将评估分数记录到日志文件中
    logging.info(f"Epoch: {epoch+1}, valid Score: {val_scores:.4f} test score: {test_scores:.4f}\n")



