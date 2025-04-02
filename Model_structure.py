import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF ## pip/conda install pytorch-crf

class FocalLoss(nn.Module):
    """
    多分类 Focal Loss，缓解长尾标签训练时的类别不平衡问题
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss *= alpha_t
        return loss.mean() if self.reduction == 'mean' else loss.sum()
        
class BertNER(BertPreTrainedModel):
    
    '''
    模型初始化
    '''

    def __init__(self, config):
        super(BertNER, self).__init__(config)  # 继承原来的BertNER
        self.num_labels = config.num_labels  #label 的数目

        self.bert = BertModel(config)  # 定义bert模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) # 随机生成weight and bias
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()
    '''
    向前传播过程
    Input: token对应的表征
    Output: 对输入token的编码
    ''' 
    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data 
        
        outputs = self.bert(input_ids, # 用bert处理
                            attention_mask=attention_mask,  
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]  # 句向量
        
        # 将原来有label的位置对应的输出提取出来
        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        
        # 将padded_sequence_output输入bilstm
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output) # 遮住一部分
        lstm_output, _ = self.bilstm(padded_sequence_output)
        
        # 进行结果的判别，返回结果
        # logits 是每个位置对有label的打分(对bilstm的输出进行维度变换)大小是(batch_size, max_len, num_labels)
        logits = self.classifier(lstm_output)  # Linear层不需要训练，只是用来改变维度的
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1) # 我们在对labels长度填充的时候,初始化值为 -1，这里是遮住填充的位置
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        '''
        Examples of (loss,) + outputs 

        作用为把loss添加到前面，作为元组的第一项

        a = (1, )
        for i in range(10):
 	        a = (2, ) + a
        print(a)
        >>(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1)
        '''
        return outputs  # loss
        
