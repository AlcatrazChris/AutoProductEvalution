import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import jieba
from torchviz import make_dot

# 数据预处理和分析
df = pd.read_csv('data/中文商品评论.csv', header=0)
df.dropna()

# 使用jieba进行中文分词
def tokenize(text):
    return ' '.join(jieba.cut(text))

df['Comment'] = df['Comment'].astype(str).apply(tokenize)

# 创建词典
word2index = {}
index2word = {}

# 添加特殊标记
special_tokens = ['<PAD>', '<UNK>']
for token in special_tokens:
    word2index[token] = len(word2index)
    index2word[len(index2word)] = token

# 构建词典
for comment in df['Comment']:
    tokens = comment.split()
    for token in tokens:
        if token not in word2index:
            word2index[token] = len(word2index)
            index2word[len(index2word)] = token

torch.save(word2index, 'model/word2index.pt')
torch.save(index2word, 'model/index2word.pt')
# 将评论转换为索引序列
def comment_to_indices(comment):
    tokens = comment.split()
    indices = []
    for token in tokens:
        if token in word2index:
            indices.append(word2index[token])
        else:
            indices.append(word2index['<UNK>'])
    return indices

df['Comment'] = df['Comment'].apply(comment_to_indices)

# 定义自定义数据集
class CommentDataset(Dataset):
    def __init__(self, df):
        self.comments = df['Comment'].tolist()
        self.labels = df['Class'].tolist()

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.comments[index]
        label = self.labels[index]
        return torch.LongTensor(comment), label

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = output[:, -1, :]
        logits = self.fc(output)
        return logits

# 定义超参数
vocab_size = len(word2index)
embedding_dim = 100
hidden_dim = 128
output_dim = 3  # 3个类别，positive、neutral、negative

# 将标签转换为从 0 开始的整数索引
label_map = {-1: 0, 0: 1, 1: 2}
df['Class'] = df['Class'].map(label_map)


# 创建数据集和数据加载器
dataset = CommentDataset(df)
batch_size = 32

# 使用pad_sequence进行填充和堆叠
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: (pad_sequence([i[0] for i in x], batch_first=True), torch.tensor([i[1] for i in x])))

# 初始化模型和优化器
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model = model.cuda()  # 将模型放在 GPU 上
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0

    for comments, labels in dataloader:
        comments = comments.cuda()
        labels = labels.cuda().long()

        optimizer.zero_grad()

        logits = model(comments)
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * comments.size(0)
        total_samples += comments.size(0)

    epoch_loss = total_loss / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'model/SA_zh.pt')

dummy_input = torch.ones(1, 100).long()
if torch.cuda.is_available():
    model = model.cpu()
out = model(dummy_input)
make_dot(out, params=dict(model.named_parameters())).render("model/LSTMModel", format="svg")
