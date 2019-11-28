import random
import math
import os
import shutil
from collections import ChainMap, deque
import time
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras.layers import Embedding, Input, Lambda
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from gensim.models import Word2Vec
from joblib import Parallel, delayed

from GEDL.code.utils import partition_dict, preprocess_nxgraph, create_alias_table, alias_sample
from GEDL.code.walker import BiasedWalker,RandomWalker




class Struc2Vec(object):
    def __init__(self, graph, walk_length=10, epohcs=100, workers=1, verbose=0, stay_prob=0.3, opt1_reduce_len=True,
                 opt2_reduce_sim_calc=True, opt3_num_layers=None, temp_path='./temp_struc2vec/', reuse=False):
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
        self.idx = list(range(len(self.idx2node)))

        self.opt1_reduce_len = opt1_reduce_len
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers

        self.resue = reuse
        self.temp_path = temp_path

        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        if not reuse:
            shutil.rmtree(self.temp_path)
            os.mkdir(self.temp_path)

        self.create_context_graph(self.opt3_num_layers, workers, verbose)
        self.prepare_biased_walk()
        self.walker = BiasedWalker(self.idx2node, self.temp_path)
        self.sentences = self.walker.simulate_walks(
            epohcs, walk_length, stay_prob, workers, verbose)

        self._embeddings = {}

    def create_context_graph(self, max_num_layers, workers=1, verbose=0, ):

        pair_distances = self._compute_structural_distance(
            max_num_layers, workers, verbose, )
        layers_adj, layers_distances = self._get_layer_rep(pair_distances)
        pd.to_pickle(layers_adj, self.temp_path + 'layers_adj.pkl')

        layers_accept, layers_alias = self._get_transition_probs(
            layers_adj, layers_distances)
        pd.to_pickle(layers_alias, self.temp_path + 'layers_alias.pkl')
        pd.to_pickle(layers_accept, self.temp_path + 'layers_accept.pkl')

    def prepare_biased_walk(self, ):

        sum_weights = {}
        sum_edges = {}
        average_weight = {}
        gamma = {}
        layer = 0
        while (os.path.exists(self.temp_path + 'norm_weights_distance-layer-' + str(layer) + '.pkl')):
            probs = pd.read_pickle(
                self.temp_path + 'norm_weights_distance-layer-' + str(layer) + '.pkl')
            for v, list_weights in probs.items():
                sum_weights.setdefault(layer, 0)
                sum_edges.setdefault(layer, 0)
                sum_weights[layer] += sum(list_weights)
                sum_edges[layer] += len(list_weights)

            average_weight[layer] = sum_weights[layer] / sum_edges[layer]

            gamma.setdefault(layer, {})

            for v, list_weights in probs.items():
                num_neighbours = 0
                for w in list_weights:
                    if (w > average_weight[layer]):
                        num_neighbours += 1
                gamma[layer][v] = num_neighbours

            layer += 1

        pd.to_pickle(average_weight, self.temp_path + 'average_weight')
        pd.to_pickle(gamma, self.temp_path + 'gamma.pkl')

    def train(self, embed_size=128, window_size=5, workers=3, iter=5):

        # pd.read_pickle(self.temp_path+'walks.pkl')
        sentences = self.sentences

        print("Learning representation...")
        model = Word2Vec(sentences, size=embed_size, window=window_size, min_count=0, hs=1, sg=1, workers=workers,
                         iter=iter)
        print("Learning representation done!")
        self.w2v_model = model

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def _compute_ordered_degreelist(self, max_num_layers):

        degreeList = {}
        vertices = self.idx  # self.g.nodes()
        for v in vertices:
            degreeList[v] = self._get_order_degreelist_node(v, max_num_layers)
        return degreeList

    def _get_order_degreelist_node(self, root, max_num_layers=None):
        if max_num_layers is None:
            max_num_layers = float('inf')

        ordered_degree_sequence_dict = {}
        visited = [False] * len(self.graph.nodes())
        queue = deque()
        level = 0
        queue.append(root)
        visited[root] = True

        while (len(queue) > 0 and level <= max_num_layers):

            count = len(queue)
            if self.opt1_reduce_len:
                degree_list = {}
            else:
                degree_list = []
            while (count > 0):

                top = queue.popleft()
                node = self.idx2node[top]
                degree = len(self.graph[node])

                if self.opt1_reduce_len:
                    degree_list[degree] = degree_list.get(degree, 0) + 1
                else:
                    degree_list.append(degree)

                for nei in self.graph[node]:
                    nei_idx = self.node2idx[nei]
                    if not visited[nei_idx]:
                        visited[nei_idx] = True
                        queue.append(nei_idx)
                count -= 1
            if self.opt1_reduce_len:
                orderd_degree_list = [(degree, freq)
                                      for degree, freq in degree_list.items()]
                orderd_degree_list.sort(key=lambda x: x[0])
            else:
                orderd_degree_list = sorted(degree_list)
            ordered_degree_sequence_dict[level] = orderd_degree_list
            level += 1

        return ordered_degree_sequence_dict

    def _compute_structural_distance(self, max_num_layers, workers=1, verbose=0, ):

        if os.path.exists(self.temp_path + 'structural_dist.pkl'):
            structural_dist = pd.read_pickle(
                self.temp_path + 'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = Struc2Vec._cost_max
            else:
                dist_func = Struc2Vec._cost

            if os.path.exists(self.temp_path + 'degreelist.pkl'):
                degreeList = pd.read_pickle(self.temp_path + 'degreelist.pkl')
            else:
                degreeList = self._compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degreeList, self.temp_path + 'degreelist.pkl')

            if self.opt2_reduce_sim_calc:
                degrees = self._create_vectors()
                degreeListsSelected = {}
                vertices = {}
                n_nodes = len(self.idx)
                for v in self.idx:  # c:list of vertex
                    nbs = Struc2Vec._get_vertices(
                        v, len(self.graph[self.idx2node[v]]), degrees, n_nodes)
                    vertices[v] = nbs  # store nbs
                    degreeListsSelected[v] = degreeList[v]  # store dist
                    for n in nbs:
                        # store dist of nbs
                        degreeListsSelected[n] = degreeList[n]
            else:
                vertices = {}
                for v in degreeList:
                    vertices[v] = [vd for vd in degreeList.keys() if vd > v]

            results = Parallel(n_jobs=workers, verbose=verbose, )(
                delayed(Struc2Vec._compute_dtw_dist)(part_list, degreeList, dist_func) for part_list in
                partition_dict(vertices, workers))
            dtw_dist = dict(ChainMap(*results))

            structural_dist = Struc2Vec._convert_dtw_struc_dist(dtw_dist)
            pd.to_pickle(structural_dist, self.temp_path +
                         'structural_dist.pkl')

        return structural_dist

    def _create_vectors(self):
        degrees = {}  # sotre v list of degree
        degrees_sorted = set()  # store degree
        G = self.graph
        for v in self.idx:
            degree = len(G[self.idx2node[v]])
            degrees_sorted.add(degree)
            if (degree not in degrees):
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        l = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if (index > 0):
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if (index < (l - 1)):
                degrees[degree]['after'] = degrees_sorted[index + 1]

        return degrees

    def _get_layer_rep(self, pair_distances):
        layer_distances = {}
        layer_adj = {}
        for v_pair, layer_dist in pair_distances.items():
            for layer, distance in layer_dist.items():
                vx = v_pair[0]
                vy = v_pair[1]

                layer_distances.setdefault(layer, {})
                layer_distances[layer][vx, vy] = distance

                layer_adj.setdefault(layer, {})
                layer_adj[layer].setdefault(vx, [])
                layer_adj[layer].setdefault(vy, [])
                layer_adj[layer][vx].append(vy)
                layer_adj[layer][vy].append(vx)

        return layer_adj, layer_distances

    def _get_transition_probs(self, layers_adj, layers_distances):
        layers_alias = {}
        layers_accept = {}

        for layer in layers_adj:

            neighbors = layers_adj[layer]
            layer_distances = layers_distances[layer]
            node_alias_dict = {}
            node_accept_dict = {}
            norm_weights = {}

            for v, neighbors in neighbors.items():
                e_list = []
                sum_w = 0.0

                for n in neighbors:
                    if (v, n) in layer_distances:
                        wd = layer_distances[v, n]
                    else:
                        wd = layer_distances[n, v]
                    w = np.exp(-float(wd))
                    e_list.append(w)
                    sum_w += w

                e_list = [x / sum_w for x in e_list]
                norm_weights[v] = e_list
                accept, alias = create_alias_table(e_list)
                node_alias_dict[v] = alias
                node_accept_dict[v] = accept

            pd.to_pickle(
                norm_weights, self.temp_path + 'norm_weights_distance-layer-' + str(layer) + '.pkl')

            layers_alias[layer] = node_alias_dict
            layers_accept[layer] = node_accept_dict

        return layers_accept, layers_alias

    @staticmethod
    def _compute_dtw_dist(part_list, degreeList, dist_func):
        dtw_dist = {}
        for v1, nbs in part_list:
            lists_v1 = degreeList[v1]  # lists_v1 :orderd degree list of v1
            for v2 in nbs:
                lists_v2 = degreeList[v2]  # lists_v1 :orderd degree list of v2
                max_layer = min(len(lists_v1), len(lists_v2))  # valid layer
                dtw_dist[v1, v2] = {}
                for layer in range(0, max_layer):
                    dist, path = fastdtw(
                        lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                    dtw_dist[v1, v2][layer] = dist
        return dtw_dist

    @staticmethod
    def _cost(a, b):
        ep = 0.5
        m = max(a, b) + ep
        mi = min(a, b) + ep
        return ((m / mi) - 1)

    @staticmethod
    def _cost_min(a, b):
        ep = 0.5
        m = max(a[0], b[0]) + ep
        mi = min(a[0], b[0]) + ep
        return ((m / mi) - 1) * min(a[1], b[1])

    @staticmethod
    def _cost_max(a, b):
        ep = 0.5
        m = max(a[0], b[0]) + ep
        mi = min(a[0], b[0]) + ep
        return ((m / mi) - 1) * max(a[1], b[1])

    @staticmethod
    def _convert_dtw_struc_dist(distances, startLayer=1):
        """

        :param distances: dict of dict
        :param startLayer:
        :return:
        """
        for vertices, layers in distances.items():
            keys_layers = sorted(layers.keys())
            startLayer = min(len(keys_layers), startLayer)
            for layer in range(0, startLayer):
                keys_layers.pop(0)

            for layer in keys_layers:
                layers[layer] += layers[layer - 1]
        return distances

    @staticmethod
    def _get_vertices(v, degree_v, degrees, n_nodes):
        a_vertices_selected = 2 * math.log(n_nodes, 2)
        vertices = []
        try:
            c_v = 0

            for v2 in degrees[degree_v]['vertices']:
                if (v != v2):
                    vertices.append(v2)  # same degree
                    c_v += 1
                    if (c_v > a_vertices_selected):
                        raise StopIteration

            if ('before' not in degrees[degree_v]):
                degree_b = -1
            else:
                degree_b = degrees[degree_v]['before']
            if ('after' not in degrees[degree_v]):
                degree_a = -1
            else:
                degree_a = degrees[degree_v]['after']
            if (degree_b == -1 and degree_a == -1):
                raise StopIteration  # not anymore v
            degree_now = Struc2Vec._verifyDegrees(degrees, degree_v, degree_a, degree_b)
            # nearest valid degree
            while True:
                for v2 in degrees[degree_now]['vertices']:
                    if (v != v2):
                        vertices.append(v2)
                        c_v += 1
                        if (c_v > a_vertices_selected):
                            raise StopIteration

                if (degree_now == degree_b):
                    if ('before' not in degrees[degree_b]):
                        degree_b = -1
                    else:
                        degree_b = degrees[degree_b]['before']
                else:
                    if ('after' not in degrees[degree_a]):
                        degree_a = -1
                    else:
                        degree_a = degrees[degree_a]['after']

                if (degree_b == -1 and degree_a == -1):
                    raise StopIteration

                degree_now = Struc2Vec._verifyDegrees(degrees, degree_v, degree_a, degree_b)
            return list(vertices)

        except StopIteration:
            return list(vertices)

    @staticmethod
    def _verifyDegrees(degrees, degree_v_root, degree_a, degree_b):

        if (degree_b == -1):
            degree_now = degree_a
        elif (degree_a == -1):
            degree_now = degree_b
        elif (abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
            degree_now = degree_b
        else:
            degree_now = degree_a

        return degree_now


class SDNE(object):
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4, ):

        self.graph = graph
        # self.g.remove_edges_from(self.g.selfloop_edges())
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)

        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        self.A, self.L = self._create_A_L(
            self.graph, self.node2idx)  # Adj Matrix,L Matrix
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embeddings = {}

    def reset_model(self, opt='adam'):

        self.model, self.emb_model = SDNE._create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1,
                                                  l2=self.nu2)
        self.model.compile(opt, [SDNE._l_2nd(self.beta), SDNE._l_1st(self.alpha)])
        self.get_embeddings()

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                    batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
                                  batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose,
                                  shuffle=False, )
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(
                        i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = self.A[index, :].todense()
                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(
                        epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def evaluate(self, ):
        return self.model.evaluate(x=self.inputs, y=self.inputs, batch_size=self.node_size)

    def get_embeddings(self):
        self._embeddings = {}
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding

        return self._embeddings

    def _create_A_L(self, graph, node2idx):
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)

            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size))

        D_data = []
        D_index = []

        for i in range(node_size):
            D_data.append(np.sum(A_[i]))
            D_index.append(i)
        D = sp.csr_matrix((D_data, (D_index, D_index)), shape=(node_size, node_size))
        L = D - A_
        return A, L

    @staticmethod
    def _l_2nd(beta):
        def loss_2nd(y_true, y_pred):
            b_ = np.ones_like(y_true)
            b_[y_true != 0] = beta
            x = K.square((y_true - y_pred) * b_)
            t = K.sum(x, axis=-1, )
            return K.mean(t)

        return loss_2nd


    @staticmethod
    def _l_1st(alpha):
        def loss_1st(y_true, y_pred):
            L = y_true
            Y = y_pred
            batch_size = tf.to_float(K.shape(L)[0])
            return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size

        return loss_1st


    @staticmethod
    def _create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
        A = Input(shape=(node_size,))
        L = Input(shape=(None,))
        fc = A
        for i in range(len(hidden_size)):
            if i == len(hidden_size) - 1:
                fc = Dense(hidden_size[i], activation='relu',
                           kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
            else:
                fc = Dense(hidden_size[i], activation='relu',
                           kernel_regularizer=l1_l2(l1, l2))(fc)
        Y = fc
        for i in reversed(range(len(hidden_size) - 1)):
            fc = Dense(hidden_size[i], activation='relu',
                       kernel_regularizer=l1_l2(l1, l2))(fc)

        A_ = Dense(node_size, 'relu', name='2nd')(fc)
        model = Model(inputs=[A, L], outputs=[A_, Y])
        emb = Model(inputs=A, outputs=Y)
        return model, emb


class Node2Vec(object):

    def __init__(self, graph, config,):
        self.config = config
        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=self.config.p, q=self.config.q, )

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            epohcs=self.config.epohcs, walk_length=self.config.walk_length, workers=self.config.workers, verbose=self.config.verbose)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


class LINE(object):
    def __init__(self, graph,config,):
        """
        :param graph:图结构
        :param config 参数文件对象
        :return 词向量模型
        """

        self.config = config
        self.order = self.config.order
        self.optimize = self.config.optimize
        if self.order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')

        self.batch_size = self.config.batch_size
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)  # 实体 与 索引 的对照
        self.use_alias = self.config.use_alias
        self.embedding_size = self.config.embedding_size # 实体向量长度
        self._embeddings = {}
        self.negative_ratio = self.config.negative_ratio
        self.node_size = graph.number_of_nodes()  # 实体数
        self.edge_size = graph.number_of_edges()  # 边数
        self.samples_per_epoch = self.edge_size * (1 + self.config.negative_ratio)  # 每个区间内训练多少个关系对
        self._get_sampling_table()  # 初始化了 节点的两个放缩映射矩阵，边的两个放缩映射矩阵
        self._reset_model()

    def _create_model(self, numNodes, embedding_size, order='second'):
        '''

        :param numNodes: 图中的节点总数
        :param embedding_size: 实体节点编码向量长度
        :param order:
        :return:
        '''

        v_i = Input(shape=(1,))
        v_j = Input(shape=(1,))

        first_emb = Embedding(numNodes, embedding_size, name='first_emb')  # 第一次编码
        second_emb = Embedding(numNodes, embedding_size, name='second_emb')  # 第二次编码
        context_emb = Embedding(numNodes, embedding_size, name='context_emb')  # 联合编码

        v_i_emb = first_emb(v_i)
        v_j_emb = first_emb(v_j)
        v_i_emb_second = second_emb(v_i)
        v_j_context_emb = context_emb(v_j)

        first = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keepdims=False), name='first_order')([v_i_emb, v_j_emb])
        second = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keepdims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

        if order == 'first':
            output_list = [first]
        elif order == 'second':
            output_list = [second]
        elif order == 'all':
            output_list = [first, second]
        else:
            output_list = [second]

        model = Model(inputs=[v_i, v_j], outputs=output_list)

        return model, {'first': first_emb, 'second': second_emb}

    def reset_training_config(self, times):
        self.steps_per_epoch = ( (self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def _reset_model(self, ):
        '''

        :param optimize: 优化器
        :return:
        '''

        # 新建模型，获得模型和向量查询字典
        self.model, self.embedding_dict = self._create_model(
            self.node_size, self.embedding_size, self.order)

        # 设置模型，优化器和损失函数
        self.model.compile(self.optimize, self._line_loss)

        # 获得批数据
        self.batch_it = self._batch_iter(self.node2idx)

    def _get_sampling_table(self):
        """
        得到所有的节点的度的概率值（放缩过），
        :return:
        """
        # create sampling table for vertex
        power = self.config.power
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)  # 每个节点的度
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]
            ] += self.graph[edge[0]][edge[1]].get('weight', 1.0)  # 得到每个实体的度（无向图中度为入度+出度）
        """
        self.graph.edges 得到在图中所有的边结构
        self.graph[实体1] 得到的是一个字典，该字典的key值是和实体1链接的其他所有实体，value是空字典（应该是用来描述该边的属性？？？）
        self.graph[实体1][实体2].get('weight', 1.0) 得到实体1 与 实体2 的边权重，设置为1（是否可以修改边权重？还是设置边属性？)
        """
        # total_sum[i] = node_degree[i]^power
        # 对每个节点的度做了一次变换，变成了他的 power 次幂，为什么要变？？？？？
        total_sum = sum([math.pow(node_degree[i], power)
                         for i in range(numNodes)])
        # 计算每个节点的度占所有节点度的比例
        norm_prob = [float(math.pow(node_degree[j], power)) /
                     total_sum for j in range(numNodes)]
        # 此时 norm_prob 是一个长度为节点个数的向量，该向量和为1，每个元素是每个节点的度占所有度的比例
        # 通过该比例可以得到每个节点在图中的重要性程度，度越大相邻节点越多，对图的贡献越大，越重要
        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        #
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.graph.edges()])  # 得到所有的边的权重值,在数值上等于numEdges
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.graph.edges()]
        # 根据边的权重得到一个边的权重向量，边的权重是1，那么该向量全是 1

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def _batch_iter(self, node2idx):
        '''
        获得随机批数量数
        :param node2idx: 实体与索引的字典
        :return: 一个元祖(),[0] 是头实体、尾实体索引，[1]是标记符号，用来区分order模式，是一个长度为批次大小的向量，当全是1是为该区间的第一抽取，-1为之后的抽取
        在同一区间内只有一组正确的头尾对，其他的头尾实体向量对，尾实体都是随机的
        '''

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]  # 得到每个边的索引对
        edge_total = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(edge_total))  # 返回随机序列，不对原list修改
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, edge_total)
        while True:
            if mod == 0:
                # 这个区间内的数据第一次被抽取
                h = []
                t = []
                for i in range(start_index, end_index):
                    # 随机的对生成序列进行替换，替换的概率是当前位置的加权之后的概率
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    # shuffle_indices 位置根据概率值进行了变化，当edge_accept 中小于平均加权概率时就会替换，大于1（大于平均值）则不替换
                    # 替换之后，小于平均概率的索引就没了，替换成了 在之前的经过放缩大于平均值饿索引
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                # 这个区间的数据第 mod次被抽取，之后的抽取头实体向量没有发生变化，单纯的变化了尾实体向量
                # 通过 node_accept，对尾实体向量进行随意采样
                sign = np.ones(len(h)) * -1
                t = []
                for i in range(len(h)):
                    t.append(alias_sample(
                        self.node_accept, self.node_alias))

            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])

            mod += 1
            mod %= mod_size
            if mod == 0:
                # 在同一区间内采样的次数达到最大时，更改采样区间
                start_index = end_index
                end_index = min(start_index + self.batch_size, edge_total)

            if start_index >= edge_total:
                # 如果更新区间之后，设置正确的区间大小，end_idnex > start_index
                # 重新设置了一遍区间，头实体向量就要更新一次
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(edge_total))
                start_index = 0
                end_index = min(start_index + self.batch_size, edge_total)

    def get_embeddings(self, ):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                        0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch,
                                        steps_per_epoch=self.steps_per_epoch,
                                        verbose=verbose)

        return hist

    def _line_loss(slef, y_true, y_pred):
        return -K.mean(K.log(K.sigmoid(y_true * y_pred)))


class DeepWalk(object):
    def __init__(self, graph,config):
        """
        该函数用来初始化deepwalk对象
        :param graph: 输入的图模型
        """
        self.config = config
        self.graph = graph # 图结构
        self.w2v_model = None # 构造词向量的模型，
        self._embeddings = {} # 词向量字典，k是实体名，v是对应的向量

        self.walker = RandomWalker( # 在deepwalk中使用的游走方法
                                    graph,
                                    p=self.config.p, # 当p q 均为 1 时深度优先，不均为1 时广度优先，该算法没有调节深度和广度之间的优劣性
                                    q=self.config.q,
                                    )
        # 生成的随机游走序列，是一个list，每个list是一个游走序列，具体生成方法在simulate_walks
        self.sentences = self.walker.simulate_walks(
                                                    epochs = self.config.epochs ,
                                                    walk_length = self.config.walk_length, # 最长游走长度
                                                    workers = self.config.workers, # CPU并行数
                                                    verbose=self.config.verbose
                                                    )

    def train(self, **kwargs):
        """
        训练词向量模型
        :param kwargs:
        :return: 训练好的词向量模型
        """

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = self.config.word2vec_min_count
        kwargs["size"] = self.config.embedding_size
        kwargs["sg"] = self.config.word2vec_sg # 1是skipgram，0是CBOW，默认是1
        kwargs["hs"] = self.config.word2vec_hs # 1是HierarchicalSoftmax，0 是NegativeSampling，默认是1
        kwargs["negative"] = self.config.word2vec_negative  # 当词向量模型为负采样时，负采样数
        kwargs["workers"] = self.config.workers
        kwargs["window"] = self.config.window_size
        kwargs["iter"] = self.config.iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        """

        :return: 返回值是一个以实体名称为key的字典，对应的键值为刚才训练得到的该key的向量
        """
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
