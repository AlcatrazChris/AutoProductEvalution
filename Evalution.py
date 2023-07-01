import json
import pandas as pd
import torch
import torch.nn as nn
import os
import jieba
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import generate_wordcloud

class CommentDataset:
    def __init__(self, comments):
        self.comments = comments

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        return torch.LongTensor(self.comments[index])


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


class CommentEvaluator:
    def __init__(self, model_path, word2index_path, index2word_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(len(torch.load(word2index_path)), embedding_dim=100, hidden_dim=128, output_dim=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.word2index = torch.load(word2index_path)
        self.index2word = torch.load(index2word_path)
        # self.max_sequence_length = 512  # assuming the max length to be 512, please modify if different

    def tokenize(self, text):
        return ' '.join(jieba.cut(text))

    def comment_to_indices(self, comment):
        tokens = comment.split()
        indices = []
        for token in tokens:
            if token in self.word2index:
                indices.append(self.word2index[token])
            else:
                indices.append(self.word2index['<UNK>'])
        return indices

    def test_model(self, test_data):
        dataloader = DataLoader(test_data, batch_size=32, shuffle=False,
                                collate_fn=lambda x: pad_sequence(x, batch_first=True))
        predictions = []
        with torch.no_grad():
            for comments in tqdm(dataloader, desc='Testing'):
                comments = comments.cuda()
                logits = self.model(comments)
                _, predicted_labels = torch.max(logits, 1)
                predictions.extend(predicted_labels.cpu().tolist())
        return predictions
    def evaluate_comments(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return None

        try:
            df = pd.read_csv(file_path, header=0)
        except pd.errors.EmptyDataError:
            print(f"No columns to parse from file {file_path}")
            return None

        if df.empty:
            print(f"No data in {file_path}")
            df['predict'] = None
            return df

        if 'rateContent' in df.columns:
            column_name = 'rateContent'
        elif '评论' in df.columns:
            column_name = '评论'
        else:
            print("没有找到列")
            return None

        df = df.dropna(subset=[column_name])
        df[column_name] = df[column_name].astype(str).apply(self.tokenize)
        texts = df[column_name].apply(self.comment_to_indices)
        texts_data = CommentDataset(texts)

        #生成词云
        # base_name = os.path.basename(file_path)
        # name_without_ext = os.path.splitext(base_name)[0]
        # wordcloud_image_path = f'temp/Wordcloud/{name_without_ext}_wordcloud.png'
        # generate_wordcloud(df[column_name].tolist(), wordcloud_image_path)
        #LSTM预测
        predictions = self.test_model(texts_data)
        df['predict'] = predictions
        return df
        # print(predictions)

    def JD_evaluate(self, data_path, comments_dir):
        data_df = pd.read_csv(data_path)
        scores = []
        for ID in data_df['ID']:
            score = self.evaluate_and_calculate(data_df, ID, comments_dir, 'JD')
            scores.append(score)
        data_df['score'] = scores
        data_df['score_normalized'] = self.normalize_scores(scores)
        data_df.to_csv(data_path, index=False)

    def TB_evaluate(self, data_path, comments_dir):
        data_df = pd.read_csv(data_path)
        scores = []
        for ID in data_df['ID']:
            score = self.evaluate_and_calculate(data_df, ID, comments_dir, 'TB')
            scores.append(score)
        data_df['score'] = scores
        data_df['score_normalized'] = self.normalize_scores(scores)
        data_df.to_csv(data_path, index=False)

    def evaluate_and_calculate(self, data_df, ID, comments_dir, platform):
        file_path = os.path.join(comments_dir, f'{ID}comments.csv')
        df = self.evaluate_comments(file_path)
        if df is None:
            print(f"正在跳过错误ID: {ID}")
            return 0
        score = self.calculate_score(df, data_df.loc[data_df['ID'] == ID, '店铺信息'].values[0], platform)
        return score

    def calculate_score(self, df, shop_info, platform='JD'):
        score = 0
        for _, row in df.iterrows():
            if row['predict'] == 2:
                score += 3
            elif row['predict'] == 1:
                score += 1
            else:
                score -= 3
        if platform == 'JD':
            shop_info = json.loads(shop_info.replace("'", "\""))
            if shop_info['标签'] == '自营':
                score += 30
            score += int(shop_info['店铺信誉']) if len(shop_info['店铺信誉'])>=2  else 50
            print('score')
        elif platform == 'TB':
            if '天猫' in shop_info:
                score += 20
        return score

    def normalize_scores(self, scores):
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized_scores

    def Interface(self, platform):

        if platform == 'taobao':
            data_path = 'temp/data/TBdata.csv'
            comments_dir = 'temp/TB_comment'
            try:
                self.TB_evaluate(data_path, comments_dir)
                print("淘宝评价成功")
            except Exception as e:
                print("淘宝评价失败:", str(e))
        elif platform == 'jingdong':
            data_path = 'temp/data/JDdata.csv'
            comments_dir = 'temp/JD_comment'
            try:
                self.JD_evaluate(data_path, comments_dir)
                print("京东评价成功")
            except Exception as e:
                print("京东评价失败:", str(e))
        else:
            print("输入有误")