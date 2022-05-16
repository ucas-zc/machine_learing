from bayesian import *


if __name__ == "__main__":
    # 测试朴素贝叶斯算法
    test_naive_bayes()
    # 交叉测试
    error_ratio = spam_test()
    print('The error ratio of spam test is %f' % error_ratio)
