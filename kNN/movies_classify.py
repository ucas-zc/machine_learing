from numpy import *
import operator
import matplotlib.pyplot as plt


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


def draw_movies_figure(data_set, classify_set, ret):
    """
    绘制电影问题的散点图
    :param data_set: 训练数据集
    :param classify_set: 分类向量
    :param ret: 分类结果
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('kNN（电影分类问题）', fontsize=8)
    ax.set_xlim(0, 120)
    ax.set_xlabel('打斗次数', fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_ylabel('接吻次数', fontsize=8)
    ax.scatter(data_set[:, 0], data_set[:, 1], s=4)
    ax.scatter(classify_set[0], classify_set[1], c='r', s=4)
    if ret == 'A':
        ax.text(80, 100, '影片分类结果：\n当前影片为爱情片', fontdict={'size': 8})
    elif ret == 'B':
        ax.text(80, 100, '影片分类结果：\n当前影片为动作片', fontdict={'size': 8})
