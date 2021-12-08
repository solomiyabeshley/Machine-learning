import time
import unittest

import numpy as np


class new_methods:

    @staticmethod
    def matrix_multiply_matrix(matrix1, matrix2):
        res_matrix = [[0 for _ in range(len(matrix1))] for _ in range(len(matrix2[0]))]
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for r in range(len(matrix1[0])):
                    res_matrix[i][j] += matrix1[i][r] * matrix2[r][j]
        return res_matrix

    @staticmethod
    def matrix_multiply_vector(matrix, vector):
        res_vector = [0 for _ in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res_vector[i] += matrix[i][j] * vector[j]
        return res_vector

    @staticmethod
    def vector_multiply_matrix(vector, matrix):
        res_vector = [0 for _ in range(len(matrix[0]))]
        for j in range(len(matrix[0])):
            for i in range(len(matrix)):
                res_vector[j] += matrix[i][j] * vector[i]
        return res_vector

    @staticmethod
    def vector_multiply_vector(vector1, vector2):
        res = 0
        for i in range(len(vector1)):
            res += vector1[i] * vector2[i]
        return res


class testing(unittest.TestCase):
    matrix1 = [
        [2, 1, 3],
        [3, 2, 1],
        [3, 2, 2],
        [1, 2, 2]
    ]
    matrix2 = [
        [2, 1, 1, 2],
        [3, 2, 2, 1],
        [3, 1, 2, 3]
    ]
    vector1 = [1, 2, 3]
    vector2 = [2, 2, 3]

    def test_matrix_multiply_matrix(self):
        res_matrix = [
            [16, 7, 10, 14],
            [15, 8, 9, 11],
            [18, 9, 11, 14],
            [14, 7, 9, 10]
        ]
        self.assertEqual(new_methods.matrix_multiply_matrix(self.matrix1, self.matrix2), res_matrix)

    def test_matrix_multiply_vector(self):
        res_vector = [13, 10, 13, 11]
        self.assertEqual(new_methods.matrix_multiply_vector(self.matrix1, self.vector1), res_vector)

    def test_vector_multiply_matrix(self):
        res_vector = [17, 8, 11, 13]
        self.assertEqual(new_methods.vector_multiply_matrix(self.vector1, self.matrix2), res_vector)

    def test_vector_multiply_vector(self):
        res = 15
        self.assertEqual(new_methods.vector_multiply_vector(self.vector1, self.vector2), res)


def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


def test_time_speed():
    a = 100
    b = 100
    matrix1 = np.random.random((a, b))
    matrix2 = np.random.random((b, a))
    vector1 = np.random.random(b)
    vector2 = np.random.random(b)
    print("Matrix [{}x{}] multiply matrix [{}x{}]: ".format(a, b, a, b))
    begining = time.time()
    new_methods.matrix_multiply_matrix(matrix1, matrix2)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nMatrix [{}x{}] multiply vector [{}]: ".format(a, b, b))
    begining = time.time()
    new_methods.matrix_multiply_vector(matrix1, vector1)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nVector [{}] multiply matrix [{}x{}]: ".format(b, b, a))
    begining = time.time()
    new_methods.vector_multiply_matrix(vector1, matrix2)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nVector [{}] multiply vector [{}]: ".format(b, b))
    begining = time.time()
    new_methods.vector_multiply_vector(vector1, vector2)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nNumpy: ")
    print("Matrix [{}x{}] multiply matrix [{}x{}]: ".format(a, b, a, b))
    begining = time.time()
    matrix1.dot(matrix2)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nMatrix [{}x{}] multiply vector [{}]: ".format(a, b, b))
    begining = time.time()
    matrix1.dot(vector1)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nVector [{}] multiply matrix [{}x{}]: ".format(b, b, a))
    begining = time.time()
    vector1.dot(matrix2)
    end = time.time()
    print("Time: {}".format(end - begining))

    print("\nVector [{}] multiply vector [{}]: ".format(b, b))
    begining = time.time()
    vector1.dot(vector2)
    end = time.time()
    print("Time: {}".format(end - begining))


def data_output():
    matrix1 = [
        [2, 1, 3],
        [3, 2, 1],
        [3, 2, 2],
        [1, 2, 2]
    ]
    matrix2 = [
        [2, 1, 1, 2],
        [3, 2, 2, 1],
        [3, 1, 2, 3]
    ]
    vector1 = [1, 2, 3]
    vector2 = [2, 2, 3]
    print("Matrix '1': ")
    print_matrix(matrix1)
    print("\nMatrix '2': ")
    print_matrix(matrix2)
    print("\nVector '1': ")
    print(vector1)
    print("\nVector '2': ")
    print(vector2)
    print("\nMatrix '1' multiply matrix '2': ")
    res_matrix = new_methods.matrix_multiply_matrix(matrix1, matrix2)
    print_matrix(res_matrix)

    print("\nMatrix '1' multiply vector '1': ")
    res_vector = new_methods.matrix_multiply_vector(matrix1, vector1)
    print(res_vector)

    print("\nVector '1' multiply matrix '2': ")
    res_vector = new_methods.vector_multiply_matrix(vector1, matrix2)
    print(res_vector)

    print("\nVector '1' multiply vector '1': ")
    res = new_methods.vector_multiply_vector(vector1, vector2)
    print(res)


def main():
    test_time_speed()
    data_output()


main()
