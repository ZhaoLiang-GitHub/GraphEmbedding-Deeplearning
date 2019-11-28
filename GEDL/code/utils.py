import numpy as np

def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1，概率向量
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    # 此时small，large记录的是 概率是否大于或小于平均概率 1/len(area_ratio)，大于1 则大于平均概率值
    # 假设此时large的个数多余small
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()  # 从后往前，拿出最后一个大于平均值的索引，最后一个小于平均值的索引
        accept[small_idx] = area_ratio_[small_idx]  # 将最后一个小于平均值的位置上设置为该小于平均值的数值
        alias[small_idx] = large_idx  # 一个小于平均值的位置上设置为进行了放缩的一个大于平均值的索引
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])  # 在area_ratio_中的最后一个大于平均值的数值变小了，变小的幅度是最后一个小于平均值
        if area_ratio_[large_idx] < 1.0:  # 对于拿出来的大于平均值，在进行判断
            small.append(large_idx)
        else:
            large.append(large_idx)

        # 一次循环之后，accept 记录的是在一个小于平均值的位置上记录值，在alias记录的是一个小于平均值的数对于哪个大于平均值数进行了放缩，记录了这个大于平均值的索引
        # 一次操作之后，small少了一个，large少了一个，然后large、small按照对应大小可能多一个，

    # 这个while，将每个大于平均值的概率进行了缩放，可能会变大可能会变小，当while结束之后
    # accpet除了0，每个位置上就是小于平均值的数值，对应索引就是该小于平均值的数值
    # 0, 0, 0.1, 0   0.1的索引2表示第2个数小于平均值，数值是0.1
    # alias除了0，其他的都是对于该位置索引所对应的小于平均值的数，对大于平均值的数进行放缩的那个大于平均值的数的索引
    # 0, 0，3， 0       3的索引是2，表示位置是2的小于平均值的数对位置是3的大于平均值的数进行了放缩，之前的放缩（假设位置1的数之前还对位置是5的数进行了放缩）的记录被覆盖了

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1  # 将现在放缩之后仍然有的每个大于平均值的，都设置为1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1  # 将每个放缩之后仍要有的小于平均值的，设置为1
    # accept 中的小于平均值的数，都对一个大于平均值的数进行了放缩，其他的都没有对其他进行过放缩
    # alias 记录的则是每个小于平均值的数对哪个大于平均值的数进行了放缩，其余的0 则是没有进行了放缩

    # accept 1,1,0.1,1
    # alias 0，0，3，0
    return accept, alias


def alias_sample(accept, alias):
    """
    在 alias 上进行随机取样
    :param accept: 向量，除了1，就是小于1的数，小于1的数意思是在之前的概率向量中该位置的小于1的概率对一个大于平均值概率进行了放缩，既该概率值使用过
    :param alias: 向量，除了1，其他的都是位置索引，和accept对应，如果accept中不是1，那么alias中对应位置就是个索引，且alias中该位置上的索引的大于平均值的数根据accept上的小于平均值的数进行了放缩
    :return: sample index
    返回值要么根据概率输出的随机索引，要么是 alias 的上索引
    所以返回值要么1，要么是个alias上的索引，要么是个随机索引
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
