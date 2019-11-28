import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import os
from GEDL.config import Config
from GEDL.code.classify import Classifier
from GEDL.code.utils import *
from GEDL.code.models import DeepWalk,LINE,Node2Vec,SDNE,Struc2Vec

class Evaluate(object):
    # 基于实体类别标签实现对实体向量的评估
    def __init__(self,embeddings,node_classify_path):
        self.embeddings = embeddings
        self.node_classify_path = node_classify_path
    def read_node_label(self,filename, skip_head=False):
        fin = open(filename, 'r')
        X = []
        Y = []
        while 1:
            if skip_head:
                fin.readline()
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split(' ')
            X.append(vec[0])
            Y.append(vec[1:])
        fin.close()
        return X, Y

    def evaluate_embeddings(self):
        """
        一个分类器函数，用来评价向量好坏，因为每个实体有对应的标签，通过向量实现多分类
        :param embeddings:
        :return:
        """
        X, Y = self.read_node_label(self.node_classify_path)
        tr_frac = 0.8
        print("Training classifier using {:.2f}% nodes...".format(
            tr_frac * 100))
        clf = Classifier(embeddings=self.embeddings, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, tr_frac)

    def plot_embeddings(self):
        """
        将实体的向量降维到2维，然后显示出来
        :param embeddings:
        :return:
        """
        X, Y = self.read_node_label(self.node_classify_path)
        emb_list = []
        for k in X:
            emb_list.append(self.embeddings[k])
        emb_list = np.array(emb_list)

        model = TSNE(n_components=2)
        node_pos = model.fit_transform(emb_list)

        color_idx = {}
        for i in range(len(X)):
            color_idx.setdefault(Y[i][0], [])
            color_idx[Y[i][0]].append(i)

        for c, idx in color_idx.items():
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
        plt.legend()
        plt.show()

    def calculate_distance(self,vector1, vector2):
            cosine_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2))) # 余弦夹角
            euclidean_distance = np.sqrt(np.sum(np.square(vector1-vector2))) # 欧式距离
            return cosine_distance

if __name__ == "__main__":

    config = Config()
    # 构建模型
    # if config.model_name.lower() == 'deepwalk':
    #     model = DeepWalk(G)
    # elif config.model_name.lower() == 'line':
    #     model = LINE(G)
    # elif config.model_name.lower() == 'struc2vec':
    #     model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
    # elif config.model_name.lower() == 'sdne':
    #     model = SDNE(G, hidden_size=[256, 128], )
    # elif config.model_name.lower() == 'node2vec':
    #     model = Node2Vec(G, walk_length=config.walk_length, epohcs=config.epohcs,p=0.25, q=4, workers=config.workers)
    # else:
    #     model = DeepWalk(G, walk_length=config.walk_length, epohcs=config.epohcs, workers=config.workers)
    #     print('参数文件中模型名称错误，采用默认的deepwalk模型')

    # 读取关系文件，构建图结构
    G = nx.read_edgelist(config.edgelist_path,  # 文件路径
                         create_using=nx.DiGraph(),  # 选择图形容器，nx.Graph()简单无向图，DiGraph()有向图，MultiGraph()可重复边的multi-graph
                         nodetype=None,  # 从节点数据转换成对应的数据格式
                         data=[('weight', int)]  # Tuples specifying dictionary key names and types for edge data
                         )
    # 训练图表征学习模型
    model = LINE(G,config)
    model.train()

    # 基于实体类别，来验证获得的词向量质量
    # embeddings = model.get_embeddings()
    # node_classify_path = '/Users/zhaoliang14/Desktop/赵亮/GEDL/data/XunYiWenYao/寻医问药category.txt'  # 实体标签地址
    # e= Evaluate(embeddings,node_classify_path)
    # e.evaluate_embeddings()
    # e.plot_embeddings()


    # 利用该模型实现对实体向量的距离计算
    # entity_dict = {}
    # f = open(node_classify_path,'r',encoding='utf-8')
    # for i in f.readlines():
    #     entity_dict[i.strip().split(' ')[0]] = i.strip().split(' ')[1]
    # max_similarity = 0
    # max_similarity_tuple = []
    # for k1, v1 in entity_dict.items():
    #     k1_label = entity_dict[k1]
    #     if k1_label == None:
    #         # print('不在实体列表内，需要人工筛查',k1)
    #         continue
    #     if k1_label != 'disease':
    #         continue  # 测试时仅仅看一个类别的内容
    #     similarity_topk = defaultdict(lambda x: 0)
    #     for k2, v2 in entity_dict.items():
    #         if k1 != k2 and entity_dict[k2] == entity_dict[k1]:
    #         #  if k1 != k2  :
    #             vector1 = embeddings[k1]
    #             vector2 = embeddings[k2]
    #             distance = e.calculate_distance(vector1, vector2)
    #             similarity_topk[k2] = distance
    #         else:
    #             similarity_topk[k2] = 0
    #     similarity_topk = sorted(similarity_topk.items(), key=lambda x: x[1], reverse=True)
    #     if similarity_topk[0][1] == 0:
    #         print(k1, '没有找到相同实体标签内的想相近实体')
    #         continue
    #     if similarity_topk[0][1] > max_similarity:
    #         max_similarity = similarity_topk[0][1]
    #         max_similarity_tuple = [k1, similarity_topk[0][0]]
    #     print(k1.strip(), '所属实体类别是{},和他最相似的实体是:'.format(k1_label))
    #     for i in range(5):
    #         print('\t', similarity_topk[i][0])
    #     print('\n\n')
    # print('在数据中最大的相似度是{}'.format(max_similarity))
    # print('在数据中最相似的两个本体是{}'.format('、'.join(max_similarity_tuple)))
