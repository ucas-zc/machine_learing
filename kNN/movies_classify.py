from numpy import *
import operator


def create_data_set():
    """
    创建数据集
    :return: group 数据集；labels 标签
    """
    group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ['A', 'A', 'A', 'B', 'B', 'B']
    return group, labels


def classify(in_x, data_set, labels, k):
    """
    使用kNN算法对输入向量进行分类
    :param in_x: 输入向量
    :param data_set: 训练数据集
    :param labels: 训练数据级标签
    :param k: k的值
    :return: 分类结果
    """
    data_set_size = data_set.shape[0]
    # 计算输入向量与训练数据集的欧式距离
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    # 对distance进行排序，并返回排序后的索引
    sorted_distance_index = distance.argsort()
    # 选择距离最小的k个点
    class_count = {}
    for index in range(k):
        vote_label = labels[sorted_distance_index[index]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 排序
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
