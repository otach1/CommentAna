import csv
import pandas as pd
import random
import torch
import os
from transformers import BertTokenizer, BertModel
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm

""""
读取评论
"""

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def read_file(file_name):
    comments_data = None
    with open(file_name, 'r', encoding='UTF-8', ) as f:
        reader = csv.reader(f)
        # 评论数据和对应的标签信息
        comments_data = [[line[0], int(line[1])] for line in reader if len(line[0]) > 0]


    random.shuffle(comments_data)
    data = pd.DataFrame(comments_data)
    same_sentence_num = data.duplicated().sum()

    if same_sentence_num > 0:
        data = data.drop_duplicates()

    f.close()

    return data


comments_data = read_file('./comments_data.csv')
# print(comments_data)

# 训练集测试集划分线
split = 0.6
split_line = int(len(comments_data) * split)

# 划分训练集与测试集，并将pandas数据类型转化为列表类型
train_comments, train_labels = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
test_comments, test_labels = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])

# print(train_comments, '\n', train_labels, '\n')
# print(test_comments, '\n', test_labels)
"""
BERTClassifier分类器模型
"""
i = 0


class BERTClassifier(nn.Module):
    # 初始化加载模型
    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        # 定义模型
        self.bert = BertModel.from_pretrained(pretrained_name)
        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)

    def forward(self, tokens_X):
        res = self.bert(**tokens_X)
        return self.mlp(res[1])


"""
评估函数
"""


def evaluate(net, comments_data, labels_data):
    sum_correct, i = 0, 0

    while i < len(comments_data):
        comments = comments_data[i: min(i + 4, len(comments_data))]
        # print(comments) 最后一行多了一个空列表
        # 最后面缩进层次少了一层查了半天
        # error: return_tensor!
        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)
        res = net(tokens_X)
        y = torch.tensor(labels_data[i: min(i + 4, len(comments_data))]).reshape(-1).to(device=device)
        # sum_correct += (res.argmax[axis=1] == y).sum()
        sum_correct += (res.argmax(axis=1) == y).sum()
        i += 8
    return sum_correct / len(comments_data)


# d2l.train_ch13()

"""
训练bert_classifier分类器
"""


def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels,
                          device, epochs):
    # 初始化模型最大精度为0.5
    max_acc = 0.5
    # 测试未训练的模型精确度
    train_acc = evaluate(net, train_comments, train_labels)
    test_acc = evaluate(net, test_comments, test_labels)
    print('--epoch', 0, '\t--train_acc', train_acc, '\t--test_acc', test_acc)

    for epoch in tqdm(range(epochs)):
        i, sum_loss = 0, 1e-6

        while i < len(train_comments):
            comments = train_comments[i: min(i + 4, len(train_comments))]
            # print(train_comments[i: min(i + 8, len(train_comments))])

            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)
            # print(tokens_X)

            res = net(tokens_X)

            y = torch.tensor(train_labels[i: min(i + 4, len(train_comments))]).reshape(-1).to(device=device)

            optimizer.zero_grad()
            l = loss(res, y)
            l.backward()
            optimizer.step()

            sum_loss += l.detach()
            i += 8

        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)

        print('\n--epoch', epoch + 1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)

        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc

            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')


device = d2l.try_gpu()

# 最终结果3分类，输出维度为3，代表概论分布
net = BERTClassifier(output_dim=3)
net = net.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels,
                      device, 5)

net = BERTClassifier(output_dim=3)
net = net.to(device)

net.load_state_dict(torch.load('./bert.parameters'))

start = 0
while start < 20:

    comment = test_comments[start]
    token_X = tokenizer(comment, padding=True, truncation=True, return_tensors='pt',).to(device)
    label = test_labels[start]
    result = net(token_X).argmax(axis=1).item()

    print(comment)

    if result == 0:
        print('预测结果: ', 0, '----》差评', end='\t')
    elif result == 1:
        print('预测结果: ', 1, '----》中评', end='\t')
    else:
        print('预测结果: ', 2, '----》好评', end='\t')

        # 输出实际结果
    if label == 0:
        print('实际结果: ', 0, '----》差评', end='\t')
    elif label == 1:
        print('实际结果: ', 1, '----》中评', end='\t')
    else:
        print('实际结果: ', 2, '----》好评', end='\t')

    if result == label:
        print('预测正确')
    else:
        print('预测错误')

    start += 1
