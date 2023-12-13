import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

_EPS = 1e-9

def _shape_check(matrix: NDArray[np.float64]) -> bool:
    """Проверяет матрицу на квадратность"""
    shape = matrix.shape
    return len(shape) == 2 and shape[0] == shape[1]


def qr_decompose(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Находит декомпозицию исходной квадратной матрицы на ортогональную и верхнюю треугольную матрицы.
    В данной реализации используется процесс Грамма-Шмидта.
    """
    if not _shape_check(matrix):
        raise ValueError("Input matrix should be square!")

    n = matrix.shape[0]
    u_matrix = np.empty((n, n))
    u_matrix[:, 0] = matrix[:, 0]
    # Q - ортогональная матрица, R - верхняя треугольная матрица
    q_matrix, r_matrix = np.empty((n, n)), np.empty((n, n))
    # Посчитаем первое значение e0 = a0 / ||a0||, где a0 - первый столбец исходной матрицы
    q_matrix[:, 0] = u_matrix[:, 0] / norm(u_matrix[:, 0])
    # Итерация по к-ву столбцов матрицы:
    for i in range(1, n):
        # считаем u(n + 1) = a(n + 1) - r(n + 1)
        u_matrix[:, i] = matrix[:, i]
        for j in range(i):
            # r(n + 1) = (a(n+1) * e1) * e1 + ... + (a(n+1) * ek) * ek
            a_n = (matrix[:, i] @ q_matrix[:, j]) * q_matrix[:, j]
            u_matrix[:, i] -= a_n
        # считаем en = un / ||un||
        q_matrix[:, i] = u_matrix[:, i] / norm(u_matrix[:, i])

    r_matrix = np.zeros((n, n))
    # Составляем верхнюю треугольную матрицу R
    for i in range(n):
        for j in range(i, n):
            r_matrix[i, j] = matrix[:, j] @ q_matrix[:, i]


    return q_matrix, r_matrix


def qr_eigenvalues(matrix: NDArray[np.float64], *, iters=50) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Находит собственные числа заданной матрицы"""
    n = matrix.shape[0]
    tmp_prev = np.copy(matrix)
    tmp_curr = np.eye(n)
    # Делаем декомпозицию заданное число итераций
    for _ in range(iters):
        q_matrix, r_matrix = qr_decompose(tmp_prev)
        tmp_prev = r_matrix @ q_matrix
        tmp_curr = tmp_curr @ q_matrix
    # Диагональ результирующей матрицы - собственные числа исходной матрицы
    # Единично-диагональная матрица, умноженная на 
    # ортогональную составляющую разложения в количество итераций - собственные векторы
    tmp_curr[:, ::2] *= (-1)
    return np.diag(tmp_prev), tmp_curr


def _iters_eigenvalue(matrix: NDArray[np.float64], vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Находит собственное значение для собственного вектора"""
    return vector.dot(matrix.dot(vector))


def power_iterations(matrix: NDArray[np.float64], *, eps=_EPS):
    """Находит наибольшее собственное число и соответсвующий
    собственный вектор матрицы методом степенных итераций."""
    if not _shape_check(matrix):
        raise ValueError("Input matrix should be square!")
    
    n = matrix.shape[0]
    # matrix = np.linalg.inv(matrix)
    eigenvector = np.ones(n) / np.sqrt(n)
    eigenvalue = _iters_eigenvalue(matrix, eigenvector)
    eigenvalue_tmp = 10 * eigenvalue / eps

    while np.abs(eigenvalue - eigenvalue_tmp) > eps:
        eigenvalue_tmp = eigenvalue
        dot_product = matrix.dot(eigenvector)
        eigenvector = dot_product / norm(dot_product)
        eigenvalue = _iters_eigenvalue(matrix, eigenvector)
    
    return eigenvalue, eigenvector


def naive_eigens(matrix: NDArray[np.float64]):
    vectors = []
    values = []
    for _ in range(matrix.shape[0]):
        eigenvalue, eigenvector = power_iterations(matrix)
        vectors.append(eigenvector)
        values.append(eigenvalue)
        matrix = matrix - eigenvalue * eigenvector @ eigenvector.transpose()
    return values, vectors
