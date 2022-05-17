import matplotlib.pyplot as plt
from numpy import *

# 梯度上升法最大迭代次数
max_cycle = 500


def load_data_set(file_name):
    """
    加载数据集
    :param file_name: 数据集文件名
    :return: 数据集矩阵，标签矩阵
    """
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(z):
    """
    定义sigmoid函数
    :param z: 函数变量
    :return: 函数结果
    """
    return 1.0 / (1 + exp(-z))


def grad_ascent(data_mat_in, label_mat_in):
    """
    梯度上升法
    :param data_mat_in: 数据集矩阵
    :param label_mat_in: 标签矩阵
    :return: 训练好的回归系数
    """
    data_set = mat(data_mat_in)
    label_mat = mat(label_mat_in).transpose()
    m, n = shape(data_set)
    # 步长
    alpha = 0.001
    # 回归系数
    weights = ones((n, 1))
    # 迭代max_cycle次数
    for index in range(max_cycle):
        sig = sigmoid(data_set * weights)
        error = label_mat - sig
        weights = weights + alpha * data_set.transpose() * error
    return weights


def stoc_grad_ascent(data_set, label_set):
    """
    随机梯度上升法
    :param data_set: 数据集
    :param label_set: 标签
    :return: 回归系数
    """
    m, n = shape(data_set)
    alpha = 0.01
    weights = ones(n)
    for index in range(m):
        sig = sigmoid(sum(data_set[index] * weights))
        error = label_set[index] - sig
        weights = weights + alpha * error * data_set[index]
    return weights


def stoc_grad_ascent_optimize(data_set, label_set, iter_size=150):
    """
    优化随机梯度上升法
    :param data_set: 数据集
    :param label_set: 标签
    :param iter_size: 迭代次数，默认150次
    :return: 回归系数
    """
    m, n = shape(data_set)
    weights = ones(n)
    # 迭代iter_size次
    for index_i in range(iter_size):
        data_index = list(range(m))
        for index_j in range(m):
            # 计算步长
            alpha = 4 / (1.0 + index_i + index_j) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            sig = sigmoid(sum(data_set[rand_index] * weights))
            error = label_set[rand_index] - sig
            weights = weights + alpha * error * data_set[rand_index]
            del(data_index[rand_index])
    return weights


def post_best_fit(weights):
    """
    绘制数据集分布及分隔线
    :param weights: 回归系数
    :return: None
    """
    # 加载数据集
    data_mat, label_mat = load_data_set('data/test_data.txt')
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    # 将两类数据集区分出来
    x_cord_1 = []
    y_cord_1 = []
    x_cord_2 = []
    y_cord_2 = []
    for index in range(n):
        if int(label_mat[index]) == 1:
            x_cord_1.append(data_arr[index, 1])
            y_cord_1.append(data_arr[index, 2])
        else:
            x_cord_2.append(data_arr[index, 1])
            y_cord_2.append(data_arr[index, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据集，一类红色标注，一类绿色标注
    ax.scatter(x_cord_1, y_cord_1, s=30, c='red', marker='s')
    ax.scatter(x_cord_2, y_cord_2, s=30, c='green')
    # 绘制分隔线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify(int_x, weights):
    """
    分类函数
    :param int_x: 输入向量
    :param weights: 回归系数
    :return: 分类结果
    """
    prob = sigmoid(sum(int_x * weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    """
    测试（从疝气病症预测病马的死亡率）
    :return: 错误率
    """
    train_fr = open('data/horse_colic_training.txt')
    test_fr = open('data/horse_colic_test.txt')
    train_set = []
    train_label_set = []
    # 构建训练数据集和标签
    for line in train_fr.readlines():
        cur_line = line.strip().split('\t')
        line_arr = []
        for index in range(21):
            line_arr.append(float(cur_line[index]))
        train_set.append(line_arr)
        train_label_set.append(float(cur_line[21]))
    # 训练回归系数
    train_weights = stoc_grad_ascent_optimize(array(train_set),
                                              train_label_set, 500)
    # 测试
    error_count = 0
    num_test_vec = 0
    for line in test_fr.readlines():
        num_test_vec += 1.0
        cur_line = line.strip().split('\t')
        line_arr = []
        for index in range(21):
            line_arr.append(float(cur_line[index]))
        if int(classify(array(line_arr), train_weights)) \
                != int(cur_line[21]):
            error_count += 1
    # 计算错误率
    error_ratio = float(error_count) / num_test_vec
    print('The error rate of this test is %f' % error_ratio)
    return error_ratio


def mutil_test():
    """
    多次测试
    :return:
    """
    num_test = 10
    error_sum = 0.0
    for index in range(num_test):
        error_sum += colic_test()
    print('After %d iterations, the average error rate is %f' %
          (num_test, error_sum / float(num_test)))
