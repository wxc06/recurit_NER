'''
@author: Xichen Wang
'''

import config
import pandas as pd
import numpy as np
import torch
import fire
import re
from data_loader import NERDataset
from Model_structure import BertNER
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from cleanco import basename

def std_word(v):
    try:
        v = basename(basename(v.lower())) #change string to lower case, then get the base name applying basename  to get rid of common suffixes
        v = re.sub(' +',' ',re.sub(r'[^\w\s]',' ',v)).strip() # get rid of punctuation
        return v
    except:
        return v

def inference(
    input_dir: str = "data/inference_test.csv",
    col_name: str = 'title', # col_name of input data
    output_dir: str = 'data/output.csv' 
):
    with open(input_dir, 'r', encoding='utf-8') as f:
        word_list = []
        label_list = []
        df = pd.read_csv(f)
        df[col_name+'_std'] = df[col_name].apply(std_word)
        df[col_name+'_std'].astype('str')
        word = list(df[col_name+'_std'])
        for i in range(len(word)):
            words = word[i].split(' ')
            labels = ['B-RES' for i in range(len(words))]
            word_list.append(words)
            label_list.append(labels)
    word = np.array(word_list)
    label = np.array(label_list) 
    dataset = NERDataset(word, label, config) #build dataset
    # eval(input('code'))
    loader = DataLoader(dataset, batch_size = config.batch_size,
                        shuffle = False, collate_fn = dataset.collate_fn)
    #使用训练好的模型
    model = BertNER.from_pretrained(config.model_dir)
    model.to(config.device)

    tokenizer = BertTokenizer.from_pretrained(config.model_dir, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])  # 记录字符
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                   token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
           # (batch_size, max_len - padding_label_len)
        data = []
        str = ','
        for item in pred_tags:
            a = str.join(item)
            data.append(a)
        df2 = pd.DataFrame(data, columns=['label'])
        df2[col_name] = df[col_name+'_std']
        df2.to_csv(output_dir, index = False)
                # === 新增 JSON 格式结构化输出 ===
        json_output = []
        for sent, tags in zip(sent_data, pred_tags):
            text = ' '.join(sent)
            entities = []
            start_idx = 0
            for word, tag in zip(sent, tags):
                end_idx = start_idx + len(word)
                if tag != 'O':
                    tag_type = tag.split('-')[-1]  # 支持B-XXX, I-XXX等格式
                    entities.append({
                        "word": word,
                        "type": tag_type,
                        "start": start_idx,
                        "end": end_idx
                    })
                start_idx = end_idx + 1  # 加1为间隔空格
            json_output.append({
                "text": text,
                "entities": entities
            })

        # 写入 json 文件
        import json
        json_output_path = output_dir.replace('.csv', '.json')
        with open(json_output_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_output, f_json, ensure_ascii=False, indent=2)
            
        if output_dir.endswith('.json'):
            json_output = []
            for tokens, labels in zip(sent_data, pred_tags):
                sentence = " ".join(tokens)
                entities = []
                current = None
                for i, label in enumerate(labels):
                    if label.startswith("B-"):
                        if current:
                            entities.append(current)
                        current = {
                            "label": label[2:], "start": i, "end": i
                        }
                    elif label.startswith("I-") and current:
                        current["end"] = i
                    else:
                        if current:
                            entities.append(current)
                            current = None
                if current:
                    entities.append(current)
                json_output.append({"text": sentence, "entities": entities})
            with open(output_dir, 'w', encoding='utf-8') as f:
                import json
                json.dump(json_output, f, ensure_ascii=False, indent=2)
            logging.info(f"🟢 JSON格式结果已写入至: {output_dir}")





if __name__ == "__main__":
    fire.Fire(inference)
    
