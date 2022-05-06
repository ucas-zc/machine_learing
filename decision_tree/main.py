from decision_tree import *


if __name__ == '__main__':
    data_set, labels = create_data_set()
    entropy = calculate_entropy(data_set)

    best_ent, best_feature = choose_best_feature_to_split(data_set)
    print(best_ent, best_feature)

    my_tree = create_decision_tree(data_set, labels)
    print(my_tree)
