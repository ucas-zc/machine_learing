from numpy import *
import operator
import matplotlib.pyplot as plt

# 数据集文件名
data_set_file = 'dataset.txt'


def file2matrix(filename):
    """
    提取数据集文件为可处理数据（矩阵）
    :param filename: 数据集文件名
    :return: 训练样本矩阵、类标签向量
    """
    # 打开文件，并按行读书文件所有内容
    fr = open(filename)
    array_lines = fr.readlines()
    lens = len(array_lines)
    # 初始化返回矩阵（可处理的数据集）
    return_mat = zeros((lens, 3))
    class_label = []
    index = 0
    # 提取数据到return_mat中
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split(',')
        return_mat[index, :] = list_from_line[0: 3]
        class_label.append(int(float(list_from_line[-1])) + 1)
        index += 1
    return return_mat, class_label


def auto_norm(data_set):
    """
    归一化处理
    :param data_set: 数据集
    :return:
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    normal_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normal_data_set = data_set - tile(min_vals, (m, 1))
    normal_data_set = normal_data_set / tile(ranges, (m, 1))
    return normal_data_set, ranges, min_vals


def draw_engagement_figure(data_set, data_labels):
    """
    可视化结果
    :param data_set: 数据集
    :param data_labels: 标签
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('kNN（网站约会问题）', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('玩视频游戏所耗时间百分比', fontsize=6)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel('每周所消费的冰淇淋公升数', fontsize=6)
    ax.scatter(data_set[:, 1], data_set[:, 2],
               4.0 * array(data_labels), 4.0 * array(data_labels))
    bx = fig.add_subplot(2, 2, 2)
    bx.set_title('kNN（网站约会问题）', fontsize=8)
    bx.set_xlim(0, 100000)
    bx.set_xlabel('每年获得的飞行常客里程数', fontsize=6)
    bx.set_ylim(0, 100)
    bx.set_ylabel('玩视频游戏所耗时间百分比', fontsize=6)
    bx.scatter(data_set[:, 0], data_set[:, 1],
               4.0 * array(data_labels), 4.0 * array(data_labels))


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


def engagement_test():
    """
    测试，计算出错误率
    :return: 错误率
    """
    # 测试集占比
    ho_ratio = 0.1
    # 提取数据集
    engagement_mat, engagement_labels = file2matrix(data_set_file)
    # 归一化处理
    norm_mat, ranges, min_vals = auto_norm(engagement_mat)
    # 获取数据集总数
    data_set_num = norm_mat.shape[0]
    # 计算测试向量总数
    test_vec_num = int(data_set_num * ho_ratio)
    # 开始测试
    error_num = 0
    for index in range(test_vec_num):
        # kNN
        classify_ret = classify(norm_mat[index, :], norm_mat[test_vec_num: data_set_num, :],
                                engagement_labels[test_vec_num: data_set_num], 3)
        print('The classify come back with %d, the real answer is %d' % (classify_ret,
              engagement_labels[index]))
        # 统计错误分类次数
        if classify_ret != engagement_labels[index]:
            error_num += 1
    error_rate = error_num / float(test_vec_num)
    print('The total error rate is %f' % error_rate)
    return error_rate


def create_data_set(data_set_num, file_name):
    """
    创建数据集
    :param data_set_num: 数据集数量
    :param file_name: 数据集文件名
    :return:
    """
    fr = open(file_name, 'w')
    for index in range(data_set_num):
        journey, game_time_ratio, ice_cream = create_data()
        string = str(journey) + ',' + str(game_time_ratio) \
                 + ',' + str(ice_cream) + '\n'
        fr.write(string)
    fr.close()


def create_data():
    """
    创建一条数据
    :return: 飞行里程数，玩游戏时间占比，冰淇淋使用量
    """
    journey = random.randint(0, 100000)
    game_time_ratio = random.randint(0, 100)
    ice_cream = random.randint(0, 200) / float(100)
    return journey, game_time_ratio, ice_cream
