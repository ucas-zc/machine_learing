# -*- coding: UTF-8 -*-

import random
from numpy import *
import re


def create_data_set():
    """
    创建数据集与标签
    :return: 数据集，标签
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    创建单词列表（不重复）
    :param data_set: 数据集
    :return: 单词列表
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_2_vec(vocab_list, input_set):
    """
    将词条转换为一个单词表相关的向量
    :param vocab_list: 单词表
    :param input_set: 词条
    :return: 与单词表相关的向量
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('The word %s is not in our vocabulary!' % word)
    return return_vec


def train_naive_bayes(train_matrix, train_category):
    """
    计算条件概率
    :param train_matrix: 训练数据矩阵
    :param train_category: 标签类别
    :return: 条件概率
    """
    # 词条总数
    num_train_docs = len(train_matrix)
    # 单词总数
    num_train_words = len(train_matrix[0])
    # 侮辱性词汇出现的概率
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_train_words)
    p1_num = ones(num_train_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for index in range(num_train_docs):
        if train_category[index] == 1:
            p1_num += train_matrix[index]
            p1_denom += sum(train_matrix[index])
        else:
            p0_num += train_matrix[index]
            p0_denom += sum(train_matrix[index])
    p0_vec = log(p0_num / p0_denom)
    p1_vec = log(p1_num / p1_denom)
    return p0_vec, p1_vec, p_abusive


def classify_naive_bayes(vec_2_classify, p0_vec, p1_vec, p_abusive):
    """
    分类器
    :param vec_2_classify: to分类的向量
    :param p0_vec:
    :param p1_vec:
    :param p_abusive: 侮辱性留言的概率
    :return: 分类结果
    """
    p_1 = sum(vec_2_classify * p1_vec) + log(p_abusive)
    p_0 = sum(vec_2_classify * p0_vec) + log(p_abusive)
    return 1 if p_1 > p_0 else 0


def test_naive_bayes():
    """
    测试贝叶斯算法
    :return:
    """
    data_set, labels = create_data_set()
    vocab_list = create_vocab_list(data_set)
    train_mat = []
    for tmp in data_set:
        train_mat.append(set_of_words_2_vec(vocab_list, tmp))
    p0_vec, p1_vec, p_abusive = train_naive_bayes(train_mat, labels)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words_2_vec(vocab_list, test_entry))
    print("['love', 'my', 'dalmation'] classify is %d" %
          classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abusive))
    test_entry = ['my', 'stupid', 'food']
    this_doc = array(set_of_words_2_vec(vocab_list, test_entry))
    print("['my', 'stupid', 'food'] classify is %d" %
          classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abusive))


def bag_of_words_2_vec(vocab_list, input_set):
    """
    将词条转换为一个单词表相关的向量(布袋)
    :param vocab_list: 单词表
    :param input_set: 词条
    :return: 与单词表相关的向量
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print('The word %s is not in our vocabulary!' % word)
    return return_vec


def text_split(big_string):
    """
    文本字符串切割
    :param big_string: 文本字符串
    :return:
    """
    list_of_tokens = re.split(r"\b[\.,\s\n\r\n]+?\b", big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    """
    垃圾邮件样本交叉测试
    :return: 错误率
    """
    # 标签列表
    class_list = []
    # 样本列表
    doc_list = []
    # 全单词列表
    full_text = []
    for index in range(1, 26):
        # 读取黑样本
        fr_1 = open('email/spam/%d.txt' % index)
        word_1_list = text_split(fr_1.read())
        class_list.append(1)
        doc_list.append(word_1_list)
        full_text.extend(word_1_list)
        fr_1.close()
        # 读取白样本
        fr_0 = open('email/ham/%d.txt' % index)
        word_0_list = text_split(fr_0.read())
        class_list.append(0)
        doc_list.append(word_0_list)
        full_text.extend(word_0_list)
        fr_0.close()
    vocab_list = create_vocab_list(doc_list)
    train_set = list(range(50))
    test_set = []
    # 构建训练集
    for index in range(10):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])
    print('test set: ', test_set)
    print('train set: ', train_set)
    train_mat = []
    train_class = []
    for doc_index in train_set:
        train_mat.append(set_of_words_2_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_naive_bayes(array(train_mat), array(train_class))
    error_count = 0
    # 测试集分类
    for doc_index in test_set:
        word_vec = set_of_words_2_vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes(array(word_vec), p0_v, p1_v, p_spam) \
                != class_list[doc_index]:
            error_count += 1
    return float(error_count) / len(test_set)
