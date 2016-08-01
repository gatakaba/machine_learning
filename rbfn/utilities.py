# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt


class PatternCoding(object):
    """PatternCodingは実数とバイナリベクトルの対応関係を保持する

    パターンは以下の条件を満たす
    * 重複が無い
    * -1と1が均等になる
    """

    def __init__(self, binary_vector_dim, division_num, reversal_num=1):
        """
        Parameters
        ----------
         binary_vector_dim : int
             バイナリベクトルの次元数
         division_num : int
             実数の分割数
         reversal_num : int , optional
             反転数
        """
        self.binary_vector_dim = binary_vector_dim
        self.division_num = division_num
        self.reversal_num = reversal_num
        self.binary_vector_table = self.make_binary_vector_table()

    def make_binary_vector_table(self):
        """make binary vector table

        Returns
        -------
        binary_vector_table : list
           listの中身は-1,1の1d array
        """

        binary_vector_table = []
        binary_vector = np.ones(self.binary_vector_dim)
        binary_vector[:int(self.binary_vector_dim / 2)] = -1
        np.random.shuffle(binary_vector)

        binary_vector_table.append(binary_vector)

        for i in range(self.division_num):
            while True:
                tmp_binary_vector = np.copy(binary_vector_table[-1])
                # select reverse index
                index1 = list(
                    np.random.choice(np.where(tmp_binary_vector == -1)[0], size=self.reversal_num, replace=False))
                index2 = list(
                    np.random.choice(np.where(tmp_binary_vector == 1)[0], size=self.reversal_num, replace=False))
                index = index1 + index2
                # reverse selected index
                tmp_binary_vector[index] *= -1

                if not any((tmp_binary_vector == x).all() for x in binary_vector_table):
                    # if tmp_binary_vector is included in binary_vector_table, add to binary_vector_table
                    binary_vector_table.append(tmp_binary_vector)
                    break
        return binary_vector_table

    def num_to_vector(self, x):
        """
        convert real number to binary vector

        Parameters
        ----------
        x : float

        Returns
        -------
        out : ndarray 
        """

        index = int(np.floor(x * self.division_num))
        return self.binary_vector_table[index]


def test_function_lienar(X):
    return np.dot(X, [1, -1])
    # return np.sign(X[:, 0] - X[:, 1] - 1)


def test_function_nonaka(X):
    if len(X.shape) == 1:
        if (X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 < 0.04:
            return 1
        else:
            return (1 + X[0]) / 2.0 * np.sin(6 * np.pi * X[0] ** 0.5 * X[1] ** 2) ** 2
    t = []
    for x in X:
        if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.04:
            t.append(1)
        else:
            t.append(
                (1 + x[0]) / 2.0 * np.sin(6 * np.pi * x[0] ** 0.5 * x[1] ** 2) ** 2)
    return np.array(t)


def donut():
    x = np.random.normal(0, 1, size=[200, 2])
    t = np.ones(200)
    t[np.where(x[:, 0] ** 2 + x[:, 1] ** 2 < 1)] = -1
    t[np.where(3 < x[:, 0] ** 2 + x[:, 1] ** 2)] = -1


if __name__ == "__main__":
    p = PatternCoding(binary_vector_dim=100, division_num=100, reversal_num=1)
    print(p.num_to_vector(0.5))
