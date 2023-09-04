import onnxruntime as ort
import psutil
from paddlenlp.transformers import AutoTokenizer
import os
import numpy as np

class OnnxHierarchical:
    def __init__(
        self,
        device = 'cpu',
        dataset_dir = './data/',
        params_path = './model/checkpoint/',
        onnx_model_path = './model/export/',
        max_seq_length = 128,
        batch_size = 32,
        label_file = 'label.txt',
        num_threads = psutil.cpu_count(logical=False)
    ):
        self.device = device
        self.dataset_dir = dataset_dir
        self.params_path = params_path
        self.onnx_model_path = onnx_model_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.label_file = label_file
        self.num_threads = num_threads
        self.threshold = 0.5 # 归类为标签的置信度阈值

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        self.predictor = ort.InferenceSession(
            os.path.join(self.onnx_model_path, 'model.onnx'),
            sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.params_path, use_fast=True)
        self.label_list = []
        with open(os.path.join(self.dataset_dir, self.label_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                self.label_list.append(line.strip())

    def __call__(self, sentence: str):
        input_data = [sentence]
        data = self.tokenizer(
            input_data,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_position_ids=False,
            return_attention_mask=False,
        )
        tokenized_data = {}
        for tokenizer_key in data:
            tokenized_data[tokenizer_key] = np.array(data[tokenizer_key], dtype="int64")
        preprocess_result = tokenized_data

        preprocess_result_batch = {}
        for tokenizer_key in preprocess_result:
            preprocess_result_batch[tokenizer_key] = [
                preprocess_result[tokenizer_key][0]
            ]

        result = self.predictor.run(None, preprocess_result_batch)[0]
        sigmoid = np.vectorize(Sigmoid)
        prob = sigmoid(result).reshape([-1])

        label = []
        for i, p in enumerate(prob):
            if p > self.threshold:
                label.append((self.label_list[i], p))

        return label
    
def Sigmoid(x):
    """
    compute sigmoid
    """
    return 1 / (1 + np.exp(-x))