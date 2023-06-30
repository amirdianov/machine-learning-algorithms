import numpy as np


class Node:
    def __init__(self):
        self.right_child = None
        self.left_child = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node = None


class DT:

    def __init__(self, type_of_task, max_depth=10, min_entropy=0, min_elem=0):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        # self.max_nb_thresholds = max_nb_thresholds
        self.root = Node()
        self.type_of_task = type_of_task

    def train(self, inputs, targets):
        value = self.__shannon_entropy(targets, len(targets)) if self.type_of_task == 'classification' \
            else self.__disp(targets)
        self.__nb_dim = inputs.shape[1]
        self.__all_dim = np.arange(self.__nb_dim)

        self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold
        self.__build_tree(inputs, targets, self.root, 1, value)

    def __get_random_axis(self):
        pass

    def __get_all_axis(self):
        pass

    def __create_term_arr(self, targets):
        """
        :param target: классы элементов обучающей выборки, дошедшие до узла
        :return: среднее значение
        np.mean(target)
        """
        if self.type_of_task == 'classification':
            result = np.array([0] * 10)
            unique, counts = np.unique(targets, return_counts=True)
            uniq_count = np.column_stack((unique, counts))
            for elem in uniq_count:
                result[int(elem[0])] += elem[1]
            return result / len(targets)
        elif self.type_of_task == 'regression':
            return np.mean(targets)

    def __generate_all_threshold(self, inputs):
        """
        :param inputs: все элементы обучающей выборки выбранной оси
        :return: все пороги, количество порогов определяется значением параметра self.max_nb_thresholds
        Использовать np.min(inputs) и np.max(inputs)
        """
        pass

    def __generate_random_threshold(self, inputs):
        """
        :param inputs: все элементы обучающей выборки(дошедшие до узла) выбранной оси
        :return: пороги, выбранные с помощью равномерного распределения.
        Количество порогов определяется значением параметра self.max_nb_thresholds
        """
        pass

    @staticmethod
    def __disp(targets):
        """
        :param targets: классы элементов обучающей выборки, дошедшие до узла
        :return: дисперсия
        np.std(arr)
        """
        return np.std(targets) ** 2

    @staticmethod
    def __shannon_entropy(targets, N):
        """
                :param targets: классы элементов обучающей выборки, дошедшие до узла
                :param N: количество элементов обучающей выборки, дошедшие до узла

                :return: энтропи/
                np.std(arr)
        """
        entropy = 0
        unique, counts = np.unique(targets, return_counts=True)
        result = np.column_stack((unique, counts))
        for element in result:
            variable = element[1] / N
            entropy += variable * np.log2(variable)
        return -entropy

    def __inf_gain(self, targets_left, targets_right, node_ent_disp, N):
        """
        :param targets_left: targets для элементов попавших в левый узел
        :param targets_right: targets для элементов попавших в правый узел
        :param node_entropy: энтропия узла-родителя
        :param N: количество элементов, дошедших до узла родителя
        :return: information gain, энтропия для левого узла, энтропия для правого узла
        ТУТ ТОЖЕ НЕ ЦИКЛОВ, используйте собственную фунцию self.__disp
        """
        if self.type_of_task == 'classification':
            ent_left = self.__shannon_entropy(targets_left, len(targets_left))
            ent_right = self.__shannon_entropy(targets_right, len(targets_right))
            return node_ent_disp - len(targets_left) / N * ent_left - \
                   len(targets_right) / N * ent_right, ent_left, ent_right
        elif self.type_of_task == 'regression':
            disp_left = 0 if len(targets_left) == 0 else self.__disp(targets_left)
            disp_right = 0 if len(targets_right) == 0 else self.__disp(targets_right)
            return node_ent_disp - len(targets_left) / N * disp_left - \
                   len(targets_right) / N * disp_right, disp_left, disp_right

    def __build_splitting_node(self, inputs, targets, entropy, N):
        df = np.hstack((inputs, targets))
        values_for_return = []
        maxim_inform_gain = 0
        for d in range(inputs.shape[1]):
            thresholds = np.unique(inputs[:, d:d + 1])
            for tr in range(len(thresholds)):
                elem = df[:, d].astype(int)
                left = df[np.where(elem <= tr)]
                right = df[np.where(elem > tr)]
                left_target, right_target = left[:, -1], right[:, -1]
                inform_gain, left_ent_or_disp, right_ent_or_disp = \
                    self.__inf_gain(left_target, right_target, entropy, N)
                if inform_gain >= maxim_inform_gain:
                    maxim_inform_gain = inform_gain
                    values_for_return = [d, tr, np.where(elem <= tr), np.where(elem > tr), left_ent_or_disp,
                                         right_ent_or_disp]
        return values_for_return

    def __build_tree(self, inputs, targets, node, depth, entropy_disp):
        N = len(targets)
        if depth >= self.max_depth or entropy_disp <= self.min_entropy or N <= self.min_elem:
            node.terminal_node = self.__create_term_arr(targets)
        else:
            ax_max, tay_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max = \
                self.__build_splitting_node(inputs, targets, entropy_disp, N)
            node.split_ind = ax_max
            node.split_val = tay_max
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[ind_left_max], targets[ind_left_max], node.left_child, depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right_max], targets[ind_right_max], node.right_child, depth + 1,
                              disp_right_max)

    def get_predictions(self, inputs):
        """
        :param inputs: вектора характеристик
        :return: предсказания целевых значений
        """
        predictions = []
        for obj in inputs:
            node = self.root
            while node.terminal_node is None:
                if obj[node.split_ind] <= node.split_val:
                    node = node.left_child
                else:
                    node = node.right_child
            if self.type_of_task == 'classification':
                predictions.append(np.argmax(node.terminal_node))
            else:
                predictions.append(node.terminal_node)
        return predictions
