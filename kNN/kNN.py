import movies_classify as mc
import engagement as eng
import matplotlib.pyplot as plt


# 主程序入口
if __name__ == '__main__':
    """ 电影问题 """
    # 获取电影问题训练数据集与标签
    movies_group, movies_labels = mc.create_data_set()
    # 画出训练数据集分布
    plt.rcParams['font.family'] = 'SimHei'
    classify_set = [18, 90]
    class_ret = mc.classify(classify_set, data_set=movies_group, labels=movies_labels, k=3)
    mc.draw_movies_figure(movies_group, classify_set, class_ret)

    """ 网站约会问题 """
    # 创建数据集
    eng.create_data_set(1000, eng.data_set_file)
    # 读取数据
    ret_mat, class_label = eng.file2matrix(eng.data_set_file)
    # 可视化数据
    eng.draw_engagement_figure(ret_mat, class_label)
    # 测试
    eng.engagement_test()
    # 分类
    eng.classify_person()

    plt.show()
