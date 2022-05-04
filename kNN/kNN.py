import movies_classify as mc
import matplotlib.pyplot as plt


# 主程序入口
if __name__ == '__main__':
    # 获取电影问题训练数据集与标签
    movies_group, movies_labels = mc.create_data_set()
    # 画出训练数据集分布
    plt.rcParams['font.family'] = 'SimHei'
    plt.title('kNN（电影分类问题）', fontsize=14)
    plt.xlim(0, 120)
    plt.xlabel('打斗次数', fontsize=10)
    plt.ylim(0, 120)
    plt.ylabel('接吻次数', fontsize=10)
    plt.scatter(movies_group[:, 0], movies_group[:, 1])
    # 对数据集[18，90]进行分类
    classify_set = [18, 90]
    plt.scatter(classify_set[0], classify_set[1])
    class_ret = mc.classify(classify_set, data_set=movies_group, labels=movies_labels, k=3)
    if class_ret == 'A':
        plt.text(90, 100, '影片分类结果：\n当前影片为爱情片')
    elif class_ret == 'B':
        plt.text(90, 100, '影片分类结果：\n当前影片为动作片')
    plt.show()
