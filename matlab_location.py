import numpy as np
import matplotlib.pyplot as plt
from Newton import NewtonK109
from Data import DataExperiment
# Координаты датчиков треугольная сетка
# ПАЭ X [cm] Y [cm]
Dd = np.array([
    [0, 0], #null
    [-1570, 1500],  # 01
    [786, 1500],  # 02
    [1, 1500],  # 03
    [-784, 1500],  # 04
    [0, 0],  # 05 его нет!!!
    [394, 750],  # 06
    [-391, 750],  # 07
    [-1176, 750],  # 08
    [-1570, 0],  # 09
    [786, 0],  # 10
    [1, 0],  # 11
    [-784, 0],  # 12
    [0, 2083],  # 13
    [0, -583],  # 14
    [1179, 750]  # 15
])

a = np.zeros((160, 11))

g = np.zeros((150, 8))
w = 3140  # ширина развертки
for k in range(150):
    d = DataExperiment(k)
    d0 = np.array([d[0], d[1], d[2], d[3]], dtype=np.int32)
    dt = np.array([d[4], d[5], d[6]], dtype=np.float64)
    amdset = np.array([d[7], d[8]], dtype=np.float64)
    if dt[2] - dt[0] > 0.001:
        x1, y1 = Dd[d0[0], 0], Dd[d0[0], 1]
        x2, y2 = Dd[d0[1], 0], Dd[d0[1], 1]
        x3, y3 = Dd[d0[2], 0], Dd[d0[2], 1]
        x4, y4 = Dd[d0[3], 0], Dd[d0[3], 1]

        xt = np.array([x1, x2, x3, x4])
        yt = np.array([y1, y2, y3, y4])

        for i in range(1, 4):
            dist1 = np.sqrt((xt[0] - xt[i]) ** 2 + (yt[0] - yt[i]) ** 2)
            g[k, 2 * i] = dist1
            dist2 = np.sqrt((w - abs(xt[0] - xt[i])) ** 2 + (yt[0] - yt[i]) ** 2)
            g[k, 2 * i + 1] = dist2

            if dist2 < dist1:
                g[k, 0] = k
                if xt[0] > xt[i]:
                    xt[i] += w
                else:
                    xt[i] -= w

        x0v = np.mean(xt[:3])
        y0v = np.mean(yt[:3])

        x0 = np.array([x0v, y0v, 300, np.sqrt((x1 - x0v) ** 2 + (y1 - y0v) ** 2) / 300])

        Res = NewtonK109(dt, x0, xt, yt)
        if Res.all() != 0:
            if g[k, 0] > 0 and Res[0] > w / 2:
                Res[0] -= w

            if g[k, 0] > 0 and Res[0] < -w / 2:
                Res[0] += w

            a[k, 0] = k
            a[k, 1] = d[0]
            a[k, 2] = d[1]
            a[k, 3] = d[2]
            a[k, 4] = d[3]
            a[k, 5] = Res[0]
            a[k, 6] = Res[1]
            a[k, 7] = Res[2]
            a[k, 8] = amdset[0]
            a[k, 9] = amdset[1]
        else:
            print('В событии ', amdset[1], 'матрица обратилась в ноль')

a = a[~np.all(a == 0, axis=1)]

plt.plot(a[:, 5], a[:, 6], 'k.')
plt.axis('equal')
plt.axis([-w / 2 - 20, w / 2 + 20, 0, 1500])


for i in range(15):
    plt.plot(Dd[i, 0], Dd[i, 1], 'bx')
    plt.axis('equal')
    plt.xlabel('x, см', fontsize=12, fontweight='bold')
    plt.ylabel('y, см', fontsize=12, fontweight='bold')
    plt.axis([-2000, 2000, -400, 1900])



plt.plot(Dd[:, 0], Dd[:, 1], 'bx')
plt.show()
