import pickle
from numpy import *
import operator
import matplotlib.pyplot as plt


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


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


def get_num_of_leafs(my_tree):
    """
    获取树的叶子结点个数
    :param my_tree: 决策树
    :return: 叶子结点个数
    """
    num = 0
    # 获取第一个key
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    # 循环遍历这颗子树上的结点
    for tmp in second_dict.keys():
        # 当前结点为元组类型，则不为叶节点，递归计算
        if type(second_dict[tmp]).__name__ == 'dict':
            num += get_num_of_leafs(second_dict[tmp])
        else:
            num += 1
    return num


def get_tree_depth(my_tree):
    """
    获取树的深度
    :param my_tree: 树深度
    :return:
    """
    depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for tmp in second_dict.keys():
        # 当前结点为元组类型，递归计算树深度
        if type(second_dict[tmp]).__name__ == 'dict':
            this_depth = 1 + get_num_of_leafs(second_dict[tmp])
        else:
            this_depth = 1
        if this_depth > depth:
            depth = this_depth
    return depth


def plot_mid_text(cnt_pt, parent_pt, mid_text):
    """
    添加节点连接信息
    :param cnt_pt: 子节点坐标
    :param parent_pt: 父节点坐标
    :param mid_text: 节点连接信息
    :return:
    """
    x_mid = (parent_pt[0] - cnt_pt[0]) / 2.0 + cnt_pt[0]
    y_mid = (parent_pt[1] - cnt_pt[1]) / 2.0 + cnt_pt[1]
    create_plot.axl.text(x_mid, y_mid, mid_text)


def create_plot(in_tree):
    """
    绘图
    :param in_tree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_of_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def plot_tree(my_tree, parent_pt, node_text):
    """
    绘制决策树的图形
    :param my_tree: 树根节点
    :param parent_pt: 父节点
    :param node_text: 节点标注
    :return:
    """
    # 获取树的叶节点个数和深度
    num_of_leafs = get_num_of_leafs(my_tree)
    first_str = list(my_tree.keys())[0]
    # 计算第一个子结点的坐标
    cnt_pt = (plot_tree.x_off + (1.0 + float(num_of_leafs)) / 2.0
              / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cnt_pt, parent_pt, node_text)
    plot_node(first_str, cnt_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    # 循环绘制
    for tmp in second_dict.keys():
        if type(second_dict[tmp]).__name__ == 'dict':
            plot_tree(second_dict[tmp], cnt_pt, str(tmp))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[tmp], (plot_tree.x_off, plot_tree.y_off),
                      cnt_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cnt_pt, str(tmp))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    绘制节点
    :param node_text: 节点标签
    :param center_pt: 子结点坐标
    :param parent_pt: 父节点坐标
    :param node_type: 节点类型信息
    :return: 
    """
    create_plot.axl.annotate(node_text, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def classify(input_tree, feat_labels, vec):
    """
    分类函数
    :param input_tree: 决策树
    :param feat_labels: 特征
    :param vec: 待分类向量
    :return:
    """
    first_node = list(input_tree.keys())[0]
    second_node = input_tree[first_node]
    feat_index = feat_labels.index(first_node)
    for key_tmp in second_node.keys():
        if vec[feat_index] == key_tmp:
            if type(second_node[key_tmp]).__name__ == 'dict':
                classify(second_node[key_tmp], feat_labels, vec)
            else:
                global class_label
                class_label = second_node[key_tmp]
    return class_label


def store_tree(my_tree, file_name):
    """
    存储决策树
    :param my_tree: 决策树
    :param file_name: 文件名
    :return: None
    """
    fw = open(file_name, 'wb')
    pickle.dump(my_tree, fw)
    fw.close()


def load_tree(file_name):
    """
    加载决策树
    :param file_name: 文件名
    :return: 决策树
    """
    fw = open(file_name, 'rb')
    my_tree = pickle.load(fw)
    fw.close()
    return my_tree
