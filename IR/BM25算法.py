import ast
import math

import pandas as pd

# 测试文本
text = '''
孩子出现肥胖症的情况。家长要通过孩子运功和健康的饮食来缓解他的症状，可以先让他做一些有氧运动，比如慢跑，爬坡，游泳等，并且饮食上孩子多吃黄瓜，胡萝卜，菠菜等，禁止孩子吃一些油炸食品和干果类食物，这些都是干热量高脂肪的食物，而且不要让孩子总是吃完就躺在床上不动，家长在治疗小儿肥胖期间如果孩子情况严重就要及时去医院在医生的指导下给孩子治疗。
'''


class BM25(object):

    def __init__(self, docs):
        self.D = len(docs)  # 总的文档数
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D  # 所有文档的平均长度
        self.docs = docs  # 所有文档
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        scores = {}
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            value = self.idf[word] * self.f[index][word] * (self.k1 + 1) / (
                    self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl))
            scores[word] = value
        return scores


if __name__ == '__main__':
    df = pd.read_csv("../data/answer_cut.csv")
    documents = [ast.literal_eval(item) for item in df["answer_cut"]]

    s = BM25(documents)
    length = len(documents)
    words_score = []
    for i, document in enumerate(documents):
        if i % 1000 == 0:
            print(i / length)
        words_score.append(s.sim(document, i))

    df["words_score"] = pd.Series(words_score)

    df.to_csv("../data/words_score.csv", index=0, encoding="utf-8")
