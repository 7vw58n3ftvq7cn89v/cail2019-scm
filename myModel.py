import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
import os




class LSTMPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMPooling, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, features):
        word_embeddings = features["token_embeddings"]
        lstm_output, _ = self.lstm(word_embeddings)
         # 修改：取最后一个时间步的输出，即 (batch_size, hidden_size)
        last_output = lstm_output[:, -1, :]
        return lstm_output


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        x = features["token_embeddings"]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        #print("SelfAttention output shape:", attn_output.shape) 

        return attn_output
    
    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, 'model_weights.pth'))

    def load(self, input_path: str):
        self.load_state_dict(torch.load(os.path.join(input_path, 'model_weights.pth'), map_location=torch.device('cpu')))

# attention
class ScmSentenceTransformer2(SentenceTransformer):
    def __init__(self, model_path, hidden_size=768):
        super().__init__()

        # 创建 Transformer 模型
        word_embedding_model = models.Transformer(model_path)

        # 创建自注意力模型
        attention_model = SelfAttention(hidden_size, hidden_size)

        # 创建池化模型
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True,
        )

        # 将模块添加到 SentenceTransformer 模型中
        self.add_module("0", word_embedding_model)
        self.add_module("1", attention_model)
        self.add_module("2", pooling_model)

    def forward(self, input_features):
        # First, pass the input features through the Transformer module (module 0)
        transformer_output = self._modules["0"](input_features)

        # Pass the transformer output through the SelfAttention module (module 1)
        self_attention_output = self._modules["1"](transformer_output)

        # Update the 'token_embeddings' key in the features dict
        transformer_output["token_embeddings"] = self_attention_output

        # Pass the updated features through the Pooling module (module 2)
        pooling_output = self._modules["2"](transformer_output)

        # Return the final output
        return pooling_output


class ScmSentenceTransformer(SentenceTransformer):
    def __init__(self, model_path, hidden_size=768):
        # 调用父类构造函数
        super().__init__()

        # 创建Transformer模型
        word_embedding_model = models.Transformer(model_path)


        # 创建句子嵌入模型
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True,
            pooling_mode_weightedmean_tokens=True,
        )


        # 创建双向LSTM层
        bi_lstm = models.LSTM(
            word_embedding_dimension=hidden_size,
            hidden_dim=hidden_size // 2,
            num_layers=2,
            bidirectional=True,
        )


        # 创建CNN层

        cnn_layer = models.CNN(
            in_word_embedding_dimension=hidden_size,  # LSTM层的输出维度
            out_channels=hidden_size,  # 输出的特征图数量
            kernel_sizes=[3],  # 卷积核大小
        )

        # 创建句子嵌入模型
        global_pooling_model = models.Pooling(
            hidden_size,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
            pooling_mode_weightedmean_tokens=False,
        )

        # 将模块添加到SentenceTransformer模型中
        self.add_module('0', word_embedding_model)
        self.add_module('1', pooling_model)
        self.add_module('2', cnn_layer)
        self.add_module('3', bi_lstm)  
        # self.add_module('4', global_pooling_model)  