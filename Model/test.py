import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import jieba
import pandas as pd

# Define the tokenizer function
def tokenize(text):
    return ' '.join(jieba.cut(text))

# Load the saved word2index and index2word dictionaries
word2index = torch.load('model/word2index.pt')
index2word = torch.load('model/index2word.pt')

# Define the CommentDataset class
class CommentDataset:
    def __init__(self, comments):
        self.comments = comments

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        return torch.LongTensor(self.comments[index])

# Define the LSTMModel class
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

# Load the saved model
model = LSTMModel(vocab_size=len(word2index), embedding_dim=100, hidden_dim=128, output_dim=3)
model.load_state_dict(torch.load('model/SA_zh.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

def comment_to_indices(comment):
    tokens = comment.split()
    indices = []
    for token in tokens:
        if token in word2index:
            indices.append(word2index[token])
        else:
            indices.append(word2index['<UNK>'])
    return indices
# Test the model
def test_model(model, test_data):
    dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=lambda x: pad_sequence(x, batch_first=True))
    predictions = []
    with torch.no_grad():
        for comments in tqdm(dataloader, desc='Testing'):
            comments = comments.cuda()
            logits = model(comments)
            _, predicted_labels = torch.max(logits, 1)
            predictions.extend(predicted_labels.cpu().tolist())
    return predictions

# Load the test data
test_df = pd.read_csv('data/data.csv', header=0)
test_df['Comment'] = test_df['Comment'].astype(str).apply(tokenize)
test_comments = test_df['Comment'].apply(comment_to_indices)
test_data = CommentDataset(test_comments)

# Test the model
predictions = test_model(model, test_data)

# Convert the predicted labels back to original labels
label_map = {0: -1, 1: 0, 2: 1}
predicted_labels = [label_map[prediction] for prediction in predictions]
test_df['Predicted_Labels'] = predicted_labels
test_df.to_csv('data/data_with_predictions.csv', index=False)
# Print the predicted labels
print(predicted_labels)