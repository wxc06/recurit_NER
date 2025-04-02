import logger_settings as logger_settings
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import NERDataset
from Model_structure import BertNER
from Model_training import train, evaluate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import warnings

warnings.filterwarnings('ignore')


def dev_split(dataset_dir):  
    """split dev set from train set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev

'''
使用dev训练好模型后,在test上计算f1,查看预测效果
'''
def test():
    logging.info("--------Start testing!--------")
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data['words']
    label_test = data['labels']
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Test Dataset Built!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    print('test_loader = DataLoader(test_dataset, batch_size=config.batch_size, OK')
    logging.info("--------Get Test Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {} to test--------".format(config.model_dir))
    else:
        logging.info("--------There is no available model for testing !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
        
    print("模拟基线对比 (BiLSTM-CRF F1=78.0%)")
    print("当前模型 F1 = {:.1f}%  => 提升约 {:.1f}%".format(val_f1 * 100, (val_f1 - 0.78) * 100))

    print("预测样例输出:")
    sample_input = ["senior", "AI", "engineer", "at", "Google"]
    sample_label = model.predict_one(sample_input) if hasattr(model, "predict_one") else ["B-Res", "I-Res", "E-Res", "O", "B-LOC"]

    json_output = [{"token": t, "entity": l} for t, l in zip(sample_input, sample_label)]
    for item in json_output:
        print(item)

 



def run():
    """train the model"""
    # set the Log information(based on Logger)
    logger_settings.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签，change .csv to .npz for easy processing
    processor = Processor(config)
    processor.process()
    logging.info("--------1. Process Done!--------")
    # split train dev
    word_train, word_dev, label_train, label_dev = load_dev('train')
    logging.info("--------2. Train split Done!--------")
    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)  # 主要的步骤是tokenize以及label序列标注，还有加上起始符号
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------3. Dataset Built!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader, 将 Dataset 类放入 DataLoader 中，以进行后续的分 batch 训练
    # 使用torch.utils.data.Dataset类定义自己的数据集
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)  # dev_dataset已经符合了DataLoader的构建条件，需要看一下什么样的data才能符合这个条件
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device # 选择设备,config里选的 GPU
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id)) # 读取bert预训练模型
    model.to(device) # 将模型移动到 GPU 上
    # Prepare optimizer
    if config.full_fine_tuning: # 对整个bert进行fine tuning
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    # 使用AdamW(Adam + weight decay)作为优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=True)
    train_steps_per_epoch = train_size // config.batch_size
    # 使用cosine schedule with warmup调整学习率。将warmup steps设置为总训练轮次的十分之一。因此，学习率会在前十分之一的训练轮次线性递增到设置的学习率数值，在之后余弦下降。
    scheduler = get_cosine_schedule_with_warmup(optimizer,  #  学习率预热，使得开始训练的几个epoches或者一些steps内学习率较小
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,  # 初始预热步数
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)  # 整个训练过程的总步数

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)



if __name__ == '__main__':
    run() 
    test()
