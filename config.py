import os
import torch

# path (多个函数都需使用)
data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'training.npz'
test_dir = data_dir + 'testing.npz'
files = ['training', 'testing']
bert_model = os.getcwd() + '/pretrained_bert'
mybert_model = os.getcwd() + '/experiments'
roberta_model = os.getcwd() + '/pretrained_bert'
model_dir = os.getcwd() + '/experiments'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/wrong_cases.txt'


# 训练集、验证集划分比例
dev_split_size = 0.1 # train.py


# 是否对整个BERT进行fine tuning
full_fine_tuning = True # train.py


# hyper-parameter
learning_rate = 2e-5 # train.py
weight_decay = 0.001 # train.py
clip_grad = 100 # Model_training.py

batch_size = 13000 # Model_structure.py;Model_training.py;train.py
epoch_num = 20 # Model_training.py;train.py
min_epoch_num = 3 # Model_training.py;train.py
patience = 0.0002 # Model_training.py
patience_num = 3 # Model_training.py

device = torch.device('cuda:1') # Model_training.py;train.py;data_loader.py


labels = ['RES', 'FUN', 'LOC', 'O'] # data_loader.py
label2id = {
    "O": 0,
    "B-RES": 1,
    "B-FUN": 2,
    "B-LOC": 3,
    "I-RES": 4,
    "I-FUN": 5,
    "I-LOC": 6,
    "E-RES": 7,
    "E-FUN": 8,
    "E-LOC": 9,
    "S-RES": 10,
    "S-FUN": 11,
    "S-LOC": 12,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}

