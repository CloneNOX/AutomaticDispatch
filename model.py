import fasttext
from utils import split_text, fixText

class MyFastText():
    def __init__(self) -> None:
        pass

    def train(self, input: str, lr: float, dim: int, epoch: int):
        self.model = fasttext.train_supervised(input=input, lr=lr, dim=dim, epoch=epoch)

    def load_model(self, path: str):
        self.model = fasttext.load_model(path)

    def predict(self, text):
        res = self.model.predict(fixText(text))
        return res

    def split_and_predict(self, text):
        label2count = {}
        splited_text = split_text(text)
        for st in splited_text:
            res = self.model.predict(fixText(st))
            pre = res[0][0]
            prob = res[1][0]
            if pre in label2count:
                label2count[pre]['count'] += 1
                label2count[pre]['prob_sum'] += prob
            else:
                label2count[pre] = {'count': 1, 'prob_sum': prob}
        max_count = 0
        max_prob_sum = 0
        for label in label2count:
            if label2count[label]['count'] > max_count or \
            (label2count[label]['count'] == max_count and label2count[label]['prob_sum'] > max_prob_sum):
                final_label = label
                max_count = label2count[label]['count']
                max_prob_sum = label2count[label]['prob_sum']

        return ([final_label], [max_prob_sum / max_count])
    
    def save(self, path: str):
        self.model.save_model(path)