"""
该类用来存储在graphembedding中所使用的参数，在之后的主文件及模型文件中参数均不需要做修改
"""
import os
class Config(object):
    def get_path(self):
        path_list = []
        for file in os.listdir(self.kg_path):
            file_path = os.path.join(self.kg_path, file)
            path_list.append(file_path)
        category_path, edgelist_path = '', ''
        for i in path_list:
            if 'category.txt' in i:
                category_path = i
            elif 'edgelist.txt' in i:
                edgelist_path = i
        if category_path == '':
            print('文件路径内没有category.txt文件')
        if edgelist_path == '':
            print('文件路径内没有edgelist.txt文件')
        return category_path, edgelist_path


    def __init__(self):
        '''
        部分参数是整个GraphEmbedding通用的参数，不管使用什么模型都需要设定，
        其余的参数是对于不同的模型需要特殊设定的，当使用该模型时需要设定，不使用时无需修改
        '''
        self.kg_path = '/Users/didi/Desktop/滴滴_赵亮/GEDL/data/XunYiWenYao'  # 保存kg文件的路径，该路径下应该有两个文件，category.txt 来描述实体\s实体标签，edgelist.txt来描述头实体\s尾实体
        self.model_name = 'deepwalk'  # 选择的模型名字，可选的是[deepwalk,line,node2vec,sdne,struc2vec],默认是deepwalk
        self.walk_length = 10  # 随机游走中每个节点游走的最大序列长度
        self.epochs = 80  # 采样总轮数，即从数据中共对所有节点进行一遍游走的轮数，共需要游走 epochs* len(nodes）次
        self.workers = 1  # 在训练模型是采用的CPU核心数，即单机多卡中的卡数
        self.iter = 5  # 随机梯度下降法中迭代的最大次数
        self.verbose = 1  # 日志显示选项
        self.word2vec_sg = 1  # 1是skipgram，0是CBOW，默认是1
        self.word2vec_hs = 1  # 1是HierarchicalSoftmax，0 是NegativeSampling，默认是1
        self.word2vec_negative = 3  # 当词向量模型的加速方法为负采样时，负采样数
        self.word2vec_min_count = None  # 训练词向量模型中的最小词频，默认为0,在图表征学习中既是在在随机游走句子中实体出现的次数
        self.embedding_size = 128  # 实体词向量长度
        self.window_size = 10  # 词向量上下文最大距离，在图谱中即是每个实体的最长依赖实体个数
        self.p = 1  #
        self.q = 1  #
        self.category_path, self.edgelist_path = self.get_path()
        self.optimize = 'adam'  # 优化器，默认为adam


        '''deepwalk'''
        pass

        '''line'''
        self.order = 'second'  # 可选'first','second','all'，默认是second
        self.negative_ratio = 5  # 在获得批量数据时，同一个区间[start_index,end_index)的数据一共需要被随机的抽取次数，当达到次数后，更新抽取区间
        self.batch_size = 1024  #
        self.use_alias = True
        self.power = 0.75  # 放缩幂次
        '''node2vec'''

        '''sdne'''


        '''struc2vec'''