from mplcursors import HoverMode
from sklearn.cluster import DBSCAN
import math as m
import numpy as np
from matplotlib import pyplot as plt
import mplcursors
from random import randint
import pandas as pd


colorsArray = ["#00FFFF", "#DC143C", "#FFB90F", "#9932CC", "#E3CF57", "#FF69B4", "#0000FF", "#A5FA2A", "#A52A2A", "#FF6103", "#7FFF00", "#EE3B3B", "#458B00",
          "#8B2323", "#6495ED",  "#C1FFC1", "#FF1493", "#FFEC8B", "#3CB371", "#FFE4B5", "#FF4500", "#7FFFD4", "#FFFF00", "#EE82EE", "#FFE7BA", "#008080", "#00FF7F"]

# имя файла с данными
#with open('треуг сетка от Игоря результаты.xlsx', 'r') as f:
    #data = np.array([[float(num) for num in line.split('\t')] for line in f])
#data = pd.read_csv(r'треуг сетка от Игоря результаты.csv')
data = np.genfromtxt(r'треуг сетка от Игоря результаты.csv', delimiter=',')
# увеличение этого количества ведет к ослаблению ограничений
cluster_size = 10


# координаты сенсоров
with open('sensors_Triangel_Hsu-Nelson_coords.txt', 'r') as f:
    sensors = np.array([[int(num) for num in line.split(' ')[:3]] for line in f])

#print(len(data))

# фильтрация по амплитуде
#data = data[data[:, 8] > 47, :]




def get_data(points, x, y):
    return points[points[:, 5] == x & points[:, 6] == y, :]


# диаметр (прямоугольника, покрывающего точки множества)
def get_radius(array):
    max_x = np.max(array[:, 0])
    min_x = np.min(array[:, 0])
    max_y = np.max(array[:, 1])
    min_y = np.min(array[:, 1])

    return ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5


# возвращает максимальную разницу скоростей и амплитуд в пределах диапазона
# выбираем N первых точек с максимальной амплитудой из всех точек в данной области
def get_max_params(points, x0, x1, y0, y1):
    res = points[(x0 <= points[:, 5]) & (points[:, 5] <= x1) & (y0 <= points[:, 6]) & (points[:, 6] <= y1), :]

    res = res[res[:, 8].argsort(), ][::-1][:cluster_size]

    print(res[:, 5:9])
    print(int(res.size / 9))
    if res.size <= 1:
        print("Not enough points!")
        return -999999, -999999, -999999

    radius = get_radius(res[:, 5:9])
    v_diff = np.max(res[:, 7]) - np.min(res[:, 7])
    a_max = res[min(res.size // 9, cluster_size) - 1, 8] # берем минимальный порог вхождения для остальных кластеров
    print(res[min(res.size // 9, cluster_size) - 1, 8])
    return radius, v_diff, a_max
# определяем параметры эпсилоны по какому-нибудь известному нам важному участку
#params = get_max_params(data, -165, -120, 170, 190) #зона для с10
#params = get_max_params(data, -229, -207, 100, 120)
#params = get_max_params(data, -21, 15, 163, 190)
#params = get_max_params(data, -185, -170, -90, -80) # k109

params = get_max_params(data,-500,-300,1400,1600)

samples = 3  # количество соседей у точки с учетом ее самой, используется в dbscan

coeff_l = 2  #
coeff_v = 5  #  весовые коэффициенты эпсилонов
coeff_a = 3  #

eps_l = params[0]  # эпсилон по радиусу множества элементов

eps_v = params[1]  # эпсилон по скорости
#eps_v = 5
max_a = params[2]  # эпсилон по амплитуде


#eps_v = 60
#max_a = 59
#eps_l = 20

EPS = coeff_l * eps_l + coeff_v * eps_v + coeff_a * max_a  # итоговый эпсилон, применяющийся в DBScan

print('EPS = %.2f, eps_l = %.2f; eps_v = %.2f; max_a = %.2f' % (EPS, params[0], params[1], params[2]))


# функция метрики многомерного расстояния между точками
# x: координаты x, y, скорость, амплитуда
def my_metric(x, y):
    d = m.sqrt(m.pow(x[0] - y[0], 2) + m.pow(x[1] - y[1], 2))
    v = m.fabs(x[2] - y[2])
    #a = m.fabs(x[3] - y[3])
    a = min(x[3], y[3])

    if (d > eps_l) or (v > eps_v) or (a < max_a):
        return EPS + 1

    return coeff_l * d + coeff_v * v + coeff_a * a

RandState = 100
cluster_num = 10
thr_d = 1100

#db = DBSCAN(eps=EPS, min_samples=samples, metric=my_metric).fit(data[:, 5:9])
db = DBSCAN(eps=EPS, min_samples=samples, metric=my_metric).fit(data[:, 5:9])
#db = sklearn.cluster.AffinityPropagation(random_state=RandState).fit(data[:, 5:9])
#db = AgglomerativeClustering(n_clusters=None, affinity=my_metric, linkage="single", distance_threshold=thr_d).fit(data[:, 5:9])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True #dbscan
#print((db.cluster_centers_))
#print(data[:, 5:9])
#print(core_samples_mask)
#print(np.argwhere(data[:, 5:9] == db.cluster_centers_[1]))
#core_samples_mask[np.where(data[:, 5:9] in db.cluster_centers_[0])] = True #kmeans
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

unique_labels = set(labels)
unique_labels.remove(-1)

def get_point_color(xy):
    global_min_a = np.min(data[:, 8])
    global_max_a = np.max(data[:, 8])
    print("MIN_A = %f, MAX_A = %f" % (global_min_a, global_max_a))

    color = [0, 0, 0, 1]
    if xy.size / 4.0 > 0:
        max_a = np.max(xy[:, 3])
        print(max_a)
        if 40 <= max_a <= 50:
            color = [0, 0.8, 0, 1]
        elif 50 < max_a <= 60:
            color = [0.8, 0.8, 0, 1]
        elif 60 < max_a:
            color = [0.8, 0, 0, 1]
    return color

#получение случайного цвета
def get_random_color():
    return '#%06X' % randint(0, 0xFFFFFF);

# plt.scatter(
#     sensors[:, 0],
#     sensors[:, 1],
#     color=[0, 0, 1, 1],
#     s=50,
#     edgecolor="k",
#     label="Sensors",
#     marker="v"
# )

#plt.annotate("X=-382", (-382, 300), (-370, 300), color='blue')
#plt.annotate("X=382", (382, 300), (340, 300), color='blue')
#plt.annotate("Y=600", (0, 600), (0, 590), color='blue')
#plt.annotate("Y=0", (0, 0), (0, 10), color='blue')
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")

plt.scatter(
    data[:, 5],
    data[:, 6],
    color=[0,0,0,1],
    s=40,
    edgecolor="k",
)

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
print(colors)
n = 0
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    # Рисуем только те кластеры, в которых до 10 элементов
    #if data[:, 5:9][class_member_mask].size / 4 <= cluster_size or k == -1:
    n += 1
    xy = data[:, 5:9][class_member_mask & core_samples_mask]
    print(xy)
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    else:
        #col = get_point_color(xy)
        if len(unique_labels) <= len(colorsArray):
            col = colorsArray[k]
        else:
            col = colors[k]


    #col = get_point_color(xy)
    # Отрисовка точек, которые являются ядром кластера
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        color=col,
        s=200,
        edgecolor="k",
    )


# для KMEANS
#     plt.scatter(
#         db.cluster_centers_[k, 0],
#         db.cluster_centers_[k, 1],
#         color=col,
#         s=200,
#         edgecolor="k",
#     )

    # Отрисовка точек, которые находятся в кластере, но не являются его ядром (шум также считается кластером)
    xy = data[:, 5:9][class_member_mask & ~core_samples_mask]


    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        color=col,
        s=40,
        edgecolor="k",
        label="Cluster {}".format(k + 1)
    )



#plt.xlim(-382, 382)
#plt.ylim(-60, 665)

mplcursors.cursor(hover=True & HoverMode.Persistent)

img = plt.imread("hsu_nelson_triangel.PNG")
#plt.imshow(img, extent=[-320, 320, -50, 370]) # подкладка с10
plt.imshow(img, extent=[-2000, 2000, -500, 2000]) #
#plt.imshow(img, extent=[-320, 320, -50, 370]) #k109
plt.title("Clusters number: %d; eps_v = %.2f; min_a = %.2f; eps_l = %.2f; EPS = %.2f" % (n, eps_v, max_a, eps_l, EPS))
#plt.title("Clusters number: %d; random_state = %i" % (n, RandState)) #для affinity propagation
#plt.title("Clusters number: %d; random_state = %i" % (cluster_num, RandState)) #для kmeans
#plt.title("Clusters number: %d; Threshold distance = %d" % (db.n_clusters_, thr_d)) #для agglo
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
plt.gca().invert_xaxis()
plt.xticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()

#print(clustering.labels_)

#unique_labels = np.unique(clustering.labels_)

#for i in unique_labels:
#    plt.scatter(data[clustering.labels_ == i, 5], data[clustering.labels_ == i, 6], label = i)

#for i in unique_labels:
#    plt.scatter(data[clustering.labels_ == i, 5], data[clustering.labels_ == i, 6])
#    for xy in data[clustering.labels_ == i, :]:
#        plt.annotate(i, (xy[5], xy[6]))

#plt.legend()
#plt.show()

