"""
该类用来存储在graphembedding中所使用的参数，在之后的主文件及模型文件中参数均不需要做修改
"""
import os
class Config(object):
    def get_path(self,path):
        path_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
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
        self.kg_path = '/Users/admin/Desktop/GraphEmbedding-deeplearning/data/XunYiWenYao'  # 保存kg文件的路径，该路径下应该有两个文件，category.txt 来描述实体\s实体标签，edgelist.txt来描述头实体\s尾实体
        self.model_name = 'deepwalk'  # 选择的模型名字，可选的是[deepwalk,line,node2vec,sdne,struc2vec],默认是deepwalk
        self.walk_length = 10  # 随机游走中每个节点游走的最大序列长度
        self.num_walks = 80  # 最大循环次数，相当于epochs，即所有数据要经过多少轮次的游走
        self.workers = 1  # 在训练模型是采用的CPU核心数，即单机多卡中的卡数
        self.embedding_size = 128  # 实体词向量长度
        self.p = 1  #
        self.q = 1  #
        self.category_path, self.edgelist_path = self.get_path(self.kg_path)

        '''deepwalk'''


        '''line'''
        self.line_order = 'second'  # 可选'first','second','all'，默认是second
        self.optimize = 'adam'  # 在line中使用的优化器，默认为adam
        self.negative_ratio = 5  # 在每个epoch内对关系对搜索的倍数，该参数大于0
        self.line_batch_size = 1024  # 在line中的批度大小


        '''node2vec'''


        '''sdne'''


        '''struc2vec'''