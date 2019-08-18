import numpy as np

def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1，概率向量
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []  # 计算概率是否大于或小于平均概率 1/len(area_ratio)
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()  # 拿出一个大于平均值，一个小于平均值
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])  # 对于大于平均值的索引变小了
        if area_ratio_[large_idx] < 1.0:  # 对于拿出来的大于平均值，在进行判断
            small.append(large_idx)
        else:
            large.append(large_idx)
    # 这个while，将每个大于平均值的概率进行了缩放，可能会变大可能会变小，当while结束之后
    # accpet该list除了0，就是小于平均值
    # alias除了0，就是大于平均值的索引

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1  # 将现在放缩之后仍然有的每个大于平均值的，都设置为1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1  # 将每个放缩之后仍要有的小于平均值的，设置为1
    return accept, alias


def alias_sample(accept, alias):
    """

    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def preprocess_nxgraph(graph):
    """
    该函数将图结构中实体映射成索引
    :param graph: 输入的图结构
    :return: node2idx 字典 key是实体名称，value是索引，idx2node 按照字典中的索引中到字典中的词
    """
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    """
    按照批次获得数据
    :param num: 一个批次内传进来的数据量
    :param workers: 并行CPU数，
    :return: 返回一个长度为workers的list，每个元素是每个worker处理的数据量
    """
    if num % workers == 0:

        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
