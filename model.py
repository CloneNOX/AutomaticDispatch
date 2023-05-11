import functools
import os
import sys
from tqdm import tqdm

import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from hierarchical.utils import preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

class myPaddleHierarchical:
    def __init__(self,
                 device = 'gpu',
                 dataset_dir = './data/',
                 params_path = './model/checkpoint/',
                 max_seq_length = 128,
                 batch_size = 32,
                 data_file = 'data.txt',
                 label_file = 'label.txt'
        ) -> None:
        self.device = device
        self.dataset_dir = dataset_dir
        self.params_path = params_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.data_file = data_file
        self.label_file = label_file

    def load_model(self, path=None):
        if path == None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.params_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.params_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)

    def predict(self, sentence: str):
        paddle.set_device(self.device)

        # 加载标签文件
        label_list = []
        label_path = os.path.join(self.dataset_dir, self.label_file)
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())

        # 加载数据集
        data_ds = load_dataset(
            read_single_sent_dataset, data_line=sentence, is_test=True, lazy=False
        )
        trans_func = functools.partial(
            preprocess_function,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            label_nums=len(label_list),
            is_test=True,
        )
        data_ds = data_ds.map(trans_func)

        # batchify dataset
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        data_batch_sampler = BatchSampler(data_ds, batch_size=1, shuffle=False)

        data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

        # predict
        results = []
        self.model.eval()
        for batch in data_data_loader:
            logits = self.model(**batch)
            probs = F.sigmoid(logits).numpy()
            for prob in probs:
                labels = []
                for i, p in enumerate(prob):
                    if p > 0.5:
                        labels.append((label_list[i], p))
                results.append(labels)
        return results[0]

    def batch_predict(self):
        """
        Predicts the data labels.
        """
        paddle.set_device(self.device)

        # 加载标签文件
        label_list = []
        label_path = os.path.join(self.dataset_dir, self.label_file)
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())

        # 加载数据集
        data_ds = load_dataset(
            read_local_dataset, path=os.path.join(self.dataset_dir, self.data_file), is_test=True, lazy=False
        )
        trans_func = functools.partial(
            preprocess_function,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            label_nums=len(label_list),
            is_test=True,
        )
        data_ds = data_ds.map(trans_func)

        # batchify dataset
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        data_batch_sampler = BatchSampler(data_ds, batch_size=self.batch_size, shuffle=False)

        data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

        # predict
        results = []
        self.model.eval()
        for batch in tqdm(data_data_loader, desc="predicting"):
            logits = self.model(**batch)
            probs = F.sigmoid(logits).numpy()
            for prob in probs:
                labels = []
                for i, p in enumerate(prob):
                    if p > 0.5:
                        labels.append((label_list[i], p))
                results.append(labels)
        return results
    
# end of myPaddleHierarchical

# 读入单条文本作为测试样例
def read_single_sent_dataset(data_line, label_list=None, is_test=False):
        data = [data_line]
        for line in data:
            if is_test:
                items = line.strip().split("\t")
                sentence = "".join(items)
                yield {"sentence": sentence}
            else:
                items = line.strip().split("\t")
                if len(items) == 0:
                    continue
                elif len(items) == 1:
                    sentence = items[0]
                    labels = []
                else:
                    sentence = "".join(items[:-1])
                    label = items[-1]
                    labels = [label_list[l] for l in label.split(",")]
                yield {"sentence": sentence, "label": labels}
