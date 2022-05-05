from numpy import *
import operator
import os, sys


def img2vec(file_path):
    """
    数字图片转换为向量
    :param file_path:
    :return:
    """
    return_vec = zeros((1, 1024))
    fr = open(file_path)
    for index_i in range(32):
        line_str = fr.readline()
        for index_j in range(32):
            return_vec[0, index_i * 32 + index_j] \
                = int(line_str[index_j])
    return return_vec


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


def hand_writing_test():
    """
    手写体测试
    :return: 错误率
    """
    # 便签向量
    hw_labels = []
    # 训练数据集构建
    # 读取训练数据目录下所有文件名
    training_file_list = os.listdir('training_data')
    training_file_num = len(training_file_list)
    # 训练数据集矩阵初始化
    training_mat = zeros((training_file_num, 1024))
    for index in range(training_file_num):
        # 取文件名
        file_name = training_file_list[index]
        # 取数字存入hw_labels[]
        digit = file_name.split('.')[0]
        hw_labels.append(int(digit.split('_')[0]))
        # 解析图片转化为向量
        training_mat[index, :] = img2vec('training_data/%s' % file_name)
    # 测试数据
    # 读取测试数据目录下所有文件名
    test_file_list = os.listdir('test_data')
    test_file_num = len(test_file_list)
    # 测试错误数量
    error_num = 0
    for index in range(test_file_num):
        # 取文件名
        file_name = test_file_list[index]
        # 提取对应数字
        file = file_name.split('.')[0]
        digit = int(file.split('_')[0])
        test_vec = img2vec('test_data/%s' % file_name)
        classify_ret = classify(test_vec, training_mat, hw_labels, 3)
        print('The classifier come back with %d, the real digit is %d'
              % (classify_ret, digit))
        if classify_ret != digit:
            error_num += 1
    error_ratio = error_num / float(test_file_num)
    return error_ratio
