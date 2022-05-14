from decision_tree import *
import copy


if __name__ == '__main__':
    # 创建数据集
    data_set, labels = create_data_set()
    labels_bak = copy.deepcopy(labels)
    # 创建决策树
    my_tree = create_decision_tree(data_set, labels_bak)
    # 绘制图形
    create_plot(my_tree)
    # 测试分类
    # 测试向量[1,0]
    ret = classify(my_tree, labels, [1, 0])
    if ret != 'no':
        print('classify [1,0] error!')
    else:
        print('classify [1,0] suc!')
    # 测试向量[1,1]
    ret = classify(my_tree, labels, [1, 1])
    if ret != 'yes':
        print('classify [1,1] error!')
    else:
        print('classify [1,1] suc!')
    # 测试向量[0]
    ret = classify(my_tree, labels, [0])
    if ret != 'no':
        print('classify [0] error!')
    else:
        print('classify [0] suc!')

    # 测试存储
    store_tree(my_tree, 'decision_tree.txt')
    ret = load_tree('decision_tree.txt')
    print(ret)
