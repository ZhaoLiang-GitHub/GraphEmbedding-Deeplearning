import itertools
import math
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange
from GEDL.code.utils import partition_num,alias_sample, create_alias_table

class RandomWalker:
    """
    该类为随机游走的实现类
    """
    def __init__(self, G, p=1, q=1):
        """
        randomwalk是一个可重复访问已访问节点的深度优先遍历
        :param G: 输入的图结构
        :param p: 概率值，用来控制从节点A到B之后，立即又可以访问节点A的概率
                    Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: 概率值，用来控制深度优先和广度优先的权重
                    In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q
    def _deepwalk_walk(self, walk_length, start_node):
        """
        深度优先搜索节点序列
        :param walk_length: 游走长度
        :param start_node: 一个随机的开始节点名字
        :return: 返回一个长度为 walk_length的list，list的开始节点为start_node，
                之后是按照深度优先搜索节点，则当从A搜索到B的时候也有可能在从B搜索到A
                当最后一个节点没有邻接节点时，不管是否已经凑够walk_length都直接返回list
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def _node2vec_walk(self, walk_length, start_node):
        """
        采用广度优先获得节点序列
        :param walk_length: 游走节点长度
        :param start_node: 随机开始节点
        :return: 返回一个长度为walk_length的list，list的开始节点为start_node，
                当最后一个节点没有邻接节点时，不管是否已经凑够walk_length都直接返回list

        """
        self.preprocess_transition_probs()
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        # print('alias_nodes',alias_nodes)

        walk = [start_node]


        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, epochs, walk_length, workers=1, verbose=0):
        """
        该函数实现了游走采样，在epohcs轮内，每一轮对所有数据进行一次游走，游走方法有两种，根据 p q 参数控制，如果是多cpu并行，总轮数和不变
        :param epohcs: 采样总轮数，相当于epochs
        :param walk_length: 节点游走长度，获得一个长度为 walk_length 的游走节点list作为上下文
        :param workers: 将模型分布在多少块cpu运行代码
        :param verbose:
        :return:
        """

        G = self.G
        nodes = list(G.nodes()) # 图上的所有节点
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(epochs, workers))
        """
        Parallel 实现了程序并行，n_jobs 为并行的CPU数量
        delayed 实现了对于函数的并行，delayed(funcion_name)参数是需要并行的名字，只有名字，参数在后面的括号里
        """

        walks = list(itertools.chain(*results)) # 将多个迭代器合并成一个

        return walks

    def _simulate_walks(self, nodes, epohcs, walk_length,):
        """

        :param nodes: 图中所有节点
        :param epohcs: 一个Walker处理的轮数，相当于epochs
        :param walk_length: 节点游走长度
        :return: 返回值是一个长度为epohcs * len(nodes) 的list1，
                list1的每个元素是一个长度为walk_length的的list2,
                list2的每一个元素是以一个随机节点为开始的游走节点序列，
        """
        walks = []
        for _ in range(epohcs):
            random.shuffle(nodes) # 随机排列list
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self._deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self._node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def _get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, epohcs, walk_length, stay_prob=0.3, workers=1, verbose=0):

        layers_adj = pd.read_pickle(self.temp_path+'layers_adj.pkl')
        layers_alias = pd.read_pickle(self.temp_path+'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path+'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path+'gamma.pkl')
        walks = []
        initialLayer = 0

        nodes = self.idx  # list(self.g.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma) for num in
            partition_num(epohcs, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, epohcs, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(epohcs):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()
            if(r < stay_prob):  # same layer
                v = BiasedWalker._chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if(r > p_moveup):
                    if(layer > initialLayer):
                        layer = layer - 1
                else:
                    if((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path

    @staticmethod
    def _chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
    
        v_list = graphs[layer][v]
    
        idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
        v = v_list[idx]
    
        return v
