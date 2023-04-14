import functools
import os
import sys
import logging
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
                 params_path = './hierarchical/checkpoint/',
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

    def load_model(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

    def predict(self):
        """
        Predicts the data labels.
        """
        paddle.set_device(self.device)
        model = AutoModelForSequenceClassification.from_pretrained(self.params_path)
        tokenizer = AutoTokenizer.from_pretrained(self.params_path)

        label_list = []
        label_path = os.path.join(self.dataset_dir, self.label_file)
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())

        data_ds = load_dataset(
            read_local_dataset, path=os.path.join(self.dataset_dir, self.data_file), is_test=True, lazy=False
        )

        trans_func = functools.partial(
            preprocess_function,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            label_nums=len(label_list),
            is_test=True,
        )

        data_ds = data_ds.map(trans_func)

        # batchify dataset
        collate_fn = DataCollatorWithPadding(tokenizer)
        data_batch_sampler = BatchSampler(data_ds, batch_size=self.batch_size, shuffle=False)

        data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

        results = []
        model.eval()
        for batch in tqdm(data_data_loader, desc="predicting"):
            logits = model(**batch)
            probs = F.sigmoid(logits
                              ).numpy()
            for prob in probs:
                labels = []
                for i, p in enumerate(prob):
                    if p > 0.5:
                        labels.append(label_list[i])
                results.append(labels)
        return results
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout