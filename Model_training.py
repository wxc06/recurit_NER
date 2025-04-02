import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config
from Model_structure import BertNER
from supplement import f1_score, bad_case
from transformers import BertTokenizer

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

'''
利用Dataloader类的实例train_loader进行分批训练(一次训练一个batch)
'''
def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # 训练
    # set model to training mode
    model.train() # 开启训练模式, 主要是对Batch Normalization 和 Dropout 层有影响。因为这两层在训练和测试时进行的操作是不同的。
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):  # 每次加载一个batch
        batch_data, batch_token_starts, batch_labels = batch_samples
        # 在Build Dataset过程中,使用了padding,为了避免self-attention关注padding部分,我们提取出这些padding位置(0),进行mask
        batch_masks = batch_data.gt(0)  # gt(0)是判断是否大于0,若小于0则是用Padding补充的
        # 前向传播,计算结果并产生 loss
        use_focal_loss = True  # 开关控制，若关闭则使用模型自带loss

        if use_focal_loss:
            outputs = model((batch_data, batch_token_starts),
                            token_type_ids=None, attention_mask=batch_masks)[0]
            logits = outputs  # (B, L, num_labels)
            active_loss = batch_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = batch_labels.view(-1)[active_loss]
            criterion = FocalLoss()
            loss = criterion(active_logits, active_labels)
        else:
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]

        train_losses += loss.item()
        # 梯度归0, 反向传播
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)  # 设置梯度截断，防止梯度爆炸或梯度消失
        # performs updates using calculated gradients
        optimizer.step()  # 要先Loss.backward()之后再用step,先清零参数空间的梯度,用了才会更新模型
        scheduler.step()  # 更新学习率
    # 返回结果
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    logging.info("Training Finished!")
    logging.info("Final dev F1: {:.4f}".format(best_val_f1))



def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    # start training,遍历epoch,调用train_epoch进行参数更新和 loss 计算
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model, mode='dev') # evaluate是自定义函数
        val_f1 = val_metrics['f1'] # 根据f1_score的变化考虑是否保存当前模型，并设置停止训练的条件，若满足条件，则停止训练。
        val_p = val_metrics['p']
        val_r = val_metrics['r']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}, precision: {}, recall: {}".format(epoch, val_metrics['loss'], val_f1, val_p, val_r))
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}, Best val loss: {}".format(best_val_f1, best_val_loss))
            break
    logging.info("Training Finished!")
import random

def evaluate(dev_loader, model, mode='dev'):
    # 测试
    # set model to evaluation mode
    model.eval() # 固定住batch normalization和dropout，防止更新
    if mode == 'test' or mode == 'dev':
        tokenizer = BertTokenizer.from_pretrained(config.model_dir, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            print('num',idx*config.batch_size)
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test'or mode == 'dev':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])  # 记录字符
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                          token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)  # Check whether the length are the same, if different, report error
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        ## Save the wrong cases for check
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
        ## We can add code for the output of the testing results here

        
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

