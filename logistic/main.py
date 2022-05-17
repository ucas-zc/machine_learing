from logistic import *


if __name__ == '__main__':
    # 加载数据集
    data_mat, label_mat = load_data_set('data/test_data.txt')
    # 随机梯度上升法
    weights = grad_ascent(data_mat, label_mat)
    # 绘制图表
    post_best_fit(weights.getA())
    # 随机梯度上升
    weights = stoc_grad_ascent(array(data_mat), label_mat)
    # 绘制图表
    post_best_fit(weights)
    # 优化随机梯度上升法
    weights = stoc_grad_ascent_optimize(array(data_mat), label_mat, 300)
    # 绘制图表
    post_best_fit(weights)
    # 示例：从疝气病症预测病马的死亡率
    mutil_test()
