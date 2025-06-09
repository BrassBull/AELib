import math as m
import numpy as np

from sklearn.cluster import DBSCAN

# Размер кластера
CLUSTER_SIZE = 10
# Количество элементов для формирования ядра
SAMPLES = 3
# Пороговое значения для DBSCAN
EPS = -1
# Коэффициенты эпсилонов
COEFF_L = 2
COEFF_V = 5
COEFF_A = 3
# Эпсилоны
epsilons = []


# Инициализурет данные для работы метода кластеризации
def init(data, x0, x1, y0, y1):
    global EPS
    global epsilons
    epsilons = get_max_params(data, x0, x1, y0, y1)
    EPS = COEFF_L * epsilons[0] + COEFF_V * epsilons[1] + COEFF_A * epsilons[2]


# диаметр окружности, покрывающей точки множества
def get_radius(points):
    max_x = np.max(points[:, 0])
    min_x = np.min(points[:, 0])
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])

    return ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5


# возвращает максимальную разницу скоростей и амплитуд в пределах диапазона
# выбираем N первых точек с максимальной амплитудой из всех точек в данной области
# возвращает эпсилоны расстояния, скорости и амплитуды
def get_max_params(points, x0, x1, y0, y1):
    res = points[(x0 <= points[:, 5]) & (points[:, 5] <= x1) & (y0 <= points[:, 6]) & (points[:, 6] <= y1), :]

    res = res[res[:, 8].argsort(), ][::-1][:CLUSTER_SIZE]

    if res.size <= 1:
        return -999999, -999999, -999999

    radius = get_radius(res[:, 5:9])
    v_diff = np.max(res[:, 7]) - np.min(res[:, 7])
    # берем минимальный порог вхождения для остальных кластеров
    a_max = res[min(res.size // 9, CLUSTER_SIZE) - 1, 8]

    return radius, v_diff, a_max


# функция метрики многомерного расстояния между точками
# x: координаты x, y, скорость, амплитуда
def dist_metric(x, y):
    l = m.sqrt(m.pow(x[0] - y[0], 2) + m.pow(x[1] - y[1], 2))
    v = m.fabs(x[2] - y[2])
    a = min(x[3], y[3])

    if (l > epsilons[0]) or (v > epsilons[1]) or (a < epsilons[2]):
        return EPS + 1

    return COEFF_L * l + COEFF_V * v + COEFF_A * a


# data - двумерный массив данных АЭ, представляющий из себя набор строк вида (i, s1, s2, s3, s4, x, y, v, a). где:
# i - индекс строки в таблице
# s1, s2, s3, s4 - упорядоченные номера датчиков зафиксировавших сигнал
# x, y - координаты источника
# v - скорость сигнала
# a - амплитуда сигнала
def clusterization(data):
    return DBSCAN(eps=EPS, min_samples=SAMPLES, metric=dist_metric).fit(data[:, 5:9])
