'''

Data preprocess, 使用BIESO标注

'''

import os
import pandas as pd
import logging
import numpy as np
import re
from cleanco import basename

class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config
    
    def std_word(self,v):
        try:
            v = basename(basename(v.lower())) #change string to lower case, then get the base name applying basename  to get rid of common suffixes
            v = re.sub(' +',' ',re.sub(r'[^\w\s]',' ',v)).strip() # get rid of punctuation
            return v
        except:
            
            return v

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files: # training / testing
            self.preprocess(file_name)

#如果数据集的input形式有变化，可以在此函数修改
    def preprocess(self,mode):
        """
        params:
            words:将文件每一行中的文本分离出来,存储为words列表
            labels:标记文本对应的标签,存储为labels
        examples:
            words示例:['senior', 'manager']
            labels示例:['B-RES', 'E-RES']
        """
        input_dir = self.data_dir + str(mode) + '.csv'
        output_dir = self.data_dir + str(mode) + '.npz' 
        if os.path.exists(output_dir) is True:  # 确认是否已经处理好了数据
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            # 先读取到内存中，然后逐行处理
            df = pd.read_csv(f)
            df['title'+'_std'] = df['title'].apply(std_word)
            df['title'+'_std'].astype('str')
            word = list(df['title'+'_std'])
            label = list(df['label'])
            for i in range(len(word)):
                words = word[i].split(' ')
                labels = label[i].split(' ')
                word_list.append(words)
                label_list.append(words)
        # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list) # 给numpy数组命名，读取的时候可以直接根据这个名字进行读取
            logging.info("--------{} data process DONE!--------".format(mode))
