from numpy import *
import operator


def create_data_set():
    """
    创建数据集和标签
    :return: 数据集、标签
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calculate_entropy(data_set):
    """
    计算信息熵
    :param data_set: 数据集
    :return: 信息熵
    """
    # 数据集总个数
    entropy_num = len(data_set)
    # 放置标签的字典
    labels_count = {}
    # 统计数据集每个类别的个数
    for tmp in data_set:
        cur_label = tmp[-1]
        if cur_label not in labels_count.keys():
            labels_count[cur_label] = 0
        labels_count[cur_label] += 1
    # 计算熵
    entropy = 0.0
    for cur_key in labels_count:
        prob = float(labels_count[cur_key]) / entropy_num
        entropy -= prob * math.log(prob, 2)
    return entropy


def split_data_set(data_set, axis, value):
    """
    划分数据集
    :param data_set: 数据集
    :param axis: 待划分属性
    :param value: 待划分属性值
    :return: 划分好的数据集
    """
    ret_data_set = []
    for tmp in data_set:
        if tmp[axis] == value:
            reduce_data_set = tmp[: axis]
            reduce_data_set.extend(tmp[axis + 1:])
            ret_data_set.append(reduce_data_set)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择划分数据集的最佳特征
    :param data_set: 数据集
    :return: 最佳信息增益，最佳划分特征
    """
    # 特征个数
    feature_num = len(data_set[0]) - 1
    # 基础信息熵
    base_entropy = calculate_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    # 循环进行每个特征的划分，查找到最佳特征划分
    for index in range(feature_num):
        # 获取当前特征所有值（重复与不重复）
        feature_list = [tmp[index] for tmp in data_set]
        unique_feature = set(feature_list)
        new_entropy = 0.0
        # 对当前特征的每个不同值进行划分，并计算其信息熵
        for value in unique_feature:
            sub_data_set = split_data_set(data_set, index, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calculate_entropy(sub_data_set)
        # 计算信息增益
        info_gain = base_entropy - new_entropy
        # 获取最佳信息增益值
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = index
    return best_info_gain, best_feature


def majority_cnt(class_list):
    """
    获取出现次数最多的类别
    :param class_list: 类别列表
    :return:
    """
    class_count = {}
    # 统计每个类别出现的次数
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 排序
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的类别
    return sorted_class_count[0][0]


def create_decision_tree(data_set, labels):
    """
    递归构建决策树
    :param data_set: 数据集
    :param labels: 标签列表
    :return:
    """
    class_list = [tmp[-1] for tmp in data_set]
    # 如果类别完全相同，停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特性时返回出现次数最多的
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择最佳特征进行划分
    best_info_gain, best_feature = choose_best_feature_to_split(data_set)
    best_label = labels[best_feature]
    # 递归构建决策树
    my_tree = {best_label: {}}
    del(labels[best_feature])
    # 获取最佳划分特征对应的特征值（重复到不重复）
    feature_values = [tmp[best_feature] for tmp in data_set]
    unique_feature = set(feature_values)
    # 对每个特征值下面的子树进行继续划分，直到不可划分为止
    for value in unique_feature:
        sub_labels = labels[:]
        my_tree[best_label][value] = create_decision_tree(split_data_set(
            data_set, best_feature, value), sub_labels)
    return my_tree
