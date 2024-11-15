import numpy as np
import matplotlib.pyplot as plt


def location(data: np.array, sensors: np.array, w: int, visual: bool = False, ax=[-2000, 2000, -400, 1900]) -> np.array:
    Dd = sensors
    l = len(data)
    a = np.zeros((l, 11))
    g = np.zeros((l, 8))
    for k in range(l):
        d = data[k, :9]
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

            x0 = np.array([x0v, y0v, 2*150, np.sqrt((x1 - x0v) ** 2 + (y1 - y0v) ** 2) / 2*150])

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
    if visual:
        visualize_location(a, sensors, w, ax)
    return a


def NewtonK109(dt, x0, xt, yt):
    """

    Args:
        dt: Array of time differences [dt1, dt2, dt3].
        x0: Initial guess for [x, y, v, t].
        xt: Array of x-coordinates of data points [x1, x2, x3, x4].
        yt: Array of y-coordinates of data points [y1, y2, y3, y4].

    Returns:
        Xk: The refined estimate of [x, y, v, t].  Returns None if the method fails to converge.
    """

    x1, x2, x3, x4 = xt
    y1, y2, y3, y4 = yt
    dt1, dt2, dt3 = dt
    x, y, v, t = x0

    Xkm = np.array([x, y, v, t])  # Vector of unknowns

    for i in range(10):  # Maximum 10 iterations
        F = np.array([
            np.sqrt((x1 - Xkm[0]) ** 2 + (y1 - Xkm[1]) ** 2) - Xkm[2] * Xkm[3],
            np.sqrt((x2 - Xkm[0]) ** 2 + (y2 - Xkm[1]) ** 2) - Xkm[2] * (Xkm[3] + dt1),
            np.sqrt((x3 - Xkm[0]) ** 2 + (y3 - Xkm[1]) ** 2) - Xkm[2] * (Xkm[3] + dt2),
            np.sqrt((x4 - Xkm[0]) ** 2 + (y4 - Xkm[1]) ** 2) - Xkm[2] * (Xkm[3] + dt3)
        ])

        Ja = np.array([
            [-(x1 - Xkm[0]) / np.sqrt((x1 - Xkm[0]) ** 2 + (y1 - Xkm[1]) ** 2),
             -(y1 - Xkm[1]) / np.sqrt((x1 - Xkm[0]) ** 2 + (y1 - Xkm[1]) ** 2),
             -Xkm[3], -Xkm[2]],
            [-(x2 - Xkm[0]) / np.sqrt((x2 - Xkm[0]) ** 2 + (y2 - Xkm[1]) ** 2),
             -(y2 - Xkm[1]) / np.sqrt((x2 - Xkm[0]) ** 2 + (y2 - Xkm[1]) ** 2),
             -(Xkm[3] + dt1), -Xkm[2]],
            [-(x3 - Xkm[0]) / np.sqrt((x3 - Xkm[0]) ** 2 + (y3 - Xkm[1]) ** 2),
             -(y3 - Xkm[1]) / np.sqrt((x3 - Xkm[0]) ** 2 + (y3 - Xkm[1]) ** 2),
             -(Xkm[3] + dt2), -Xkm[2]],
            [-(x4 - Xkm[0]) / np.sqrt((x4 - Xkm[0]) ** 2 + (y4 - Xkm[1]) ** 2),
             -(y4 - Xkm[1]) / np.sqrt((x4 - Xkm[0]) ** 2 + (y4 - Xkm[1]) ** 2),
             -(Xkm[3] + dt3), -Xkm[2]]
        ])
        try:
            Jac = np.linalg.pinv(Ja)  # Inverse of the Jacobian matrix
            dXk = -Jac @ F  # Corrections to Xkm
            Xk = Xkm + dXk  # New approximation
            Xkm = Xk
            if np.max(np.abs(dXk)) < 1e-1:
                break  # Convergence criterion

        except:
            return (np.zeros(4))
    return Xk


def visualize_location(data: np.array, sensors: np.array, w: int, ax):
    a = data
    Dd = sensors
    plt.plot(a[:, 5], a[:, 6], 'k.')
    plt.axis('equal')
    plt.axis([-w / 2 - 20, w / 2 + 20, 0, 1500])

    for i in range(len(Dd)):
        plt.plot(Dd[i, 0], Dd[i, 1], 'bx')
        plt.axis('equal')
        plt.xlabel('x, см', fontsize=12, fontweight='bold')
        plt.ylabel('y, см', fontsize=12, fontweight='bold')
        plt.axis(ax)

    plt.plot(Dd[:, 0], Dd[:, 1], 'bx')
    plt.show()
