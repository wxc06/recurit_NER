'''
Build Dataset
Bert预训练的 Input

'''

import torch
import numpy as np
from transformers import BertTokenizer,BertTokenizerFast
from torch.utils.data import Dataset
import pandas as pd
import datasets 

class NERDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True) 
        #数据本身为list中的单个字符，所以此处不进行分词。tokenizer作用为大写转小写
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)  # 构建数据集，将每个word转换成tokenizer中的id，并且将label与id对应
        self.word_pad_idx = word_pad_idx  # 起初始化作用的
        self.label_pad_idx = label_pad_idx  # 起初始化作用的
        self.device = config.device


    def preprocess(self, origin_sentences, origin_labels):
        """
        函数功能：
        ·在每句话前面加一个开头CLS
        ·将原始字符/字都转换成id,并存储有label的字的开始位置的索引
        ·将label转成成 id
        注意：代码中 token 的长度都是 1,这是由 .npz 中的数据作为输入决定的
   
        examples: 
            word:['[CLS]', 'manager', 'senior', 'intern', 'engineer']
            sentence:([101, 3851, 1555, 7213, 6121],
                        array([ 1,  2,  3,  4]))
            label:[3, 13, 13, 13]
        """
        data = []
        sentences = []
        labels = []
        # print(origin_sentences)
        for line in origin_sentences:  # 多行的文本，遍历每一行的sentence
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:  # 一句话中的每个词
                token_result = self.tokenizer.tokenize(token)  
                if len(token_result) > 1:
                    words.append([token_result[0]])  # 用bert tokenizer把字符转换成id ***
                else:
                    words.append(token_result)
                word_lens.append(1) 
            # 变成单个字的列表，开头加上[CLS],CLS没有特别的语义，主要为了判断两个句子是否连接在一起
            words = ['[CLS]'] + [item for token in words for item in token]  # token 是字符列表, item 是 token 中的项
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])  # 除了 `[CLS]` 之外的索引, 写成一个列表
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))  # 将 token 的 id 和 index 一起加入 setences
        
        for tag in origin_labels:  # tag 是每一行的 origin_sentences 中的字对应的 label
            label_id = [self.label2id.get(t) for t in tag]  # 每个字的 label -> id
            labels.append(label_id)
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data  # 作为 self.dataset

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]  # 记录了text每个字母对应的index以及每个字母的起始位置
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度(batch中最长的data长度)
            2. aligning: 找到每个sentence sequence里面有label项,文本与label对齐
            3. tensor:转化为tensor
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = 0
        for s in sentences:  # 找到这个batch中最长句子的长度
            max_len = max(len(s[0]), max_len)

        # padding data 初始化矩阵（句子个数, 最大句子长度）
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))  # 初始化batch*length的数据
        batch_label_starts = []
        max_label_len = 0

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)
        
        # max_label_len = max_len ##

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]



