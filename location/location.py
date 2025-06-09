WIDTH : float


def get_wave(data, num):
    return data.loc[data['Num'] == str(num)]


def get_sensor_coords(coords, num):
    return coords.loc[coords['Num'] == str(num)]


# функция расстояния, учитывающая возможность прохождения волны через края объекта
def dist(x0, y0, x1, y1):
    global WIDTH
    xsqr = (x1 - x0) ** 2
    ysqr = (y1 - y0) ** 2

    d1 = (xsqr + ysqr) ** 0.5
    d2 = ((WIDTH - xsqr ** 0.5) ** 2 + ysqr) ** 0.5
    return min(d1, d2)


# уравнение скорости
def speed_eq(x, y, x1, y1, x2, y2, t1, t2):
    return abs(dist(x, y, x1, y1) - dist(x, y, x2, y2)) / (t2 - t1)


# получение списка координат первых N датчиков
def take_channels_coords(data, coords, N):
    ch_coords = [[float] * 2] * N

    for i in range(0, N):
        ch_coords[i] = [coords["x"][int(data["Channel"][i]) - 1], coords["y"][int(data["Channel"][i]) - 1]]

    return ch_coords


# проверка на правильный порядок удаленности точки (х, у) от всех точек из coords
def check_dist(x, y, coords):
    for i in range(0, len(coords) - 1):
        if dist(x, y, coords[i][0], coords[i][1]) > dist(x, y, coords[i + 1][0], coords[i + 1][1]):
            return False
    return True


# локация источников для одной волны
def localize_wave(data, coords, lthreshold, uthreshold, events_n, acc):
    res = []
    ch_coords = take_channels_coords(data, coords, events_n)

    diag = dist(ch_coords[0][0], ch_coords[0][1], ch_coords[1][0], ch_coords[1][1])

    xi_min = ch_coords[0][0] - diag / 2
    xi_max = ch_coords[0][0] + diag / 2

    yi_min = ch_coords[0][1] - diag / 2
    yi_max = ch_coords[0][1] + diag / 2

    xi = xi_min
    yi = yi_min

    s1 = speed_eq(xi, yi, ch_coords[0][0], ch_coords[0][1], ch_coords[1][0], ch_coords[1][1],
                  float(data["Time"][0]), float(data["Time"][1]))
    s2 = speed_eq(xi, yi, ch_coords[1][0], ch_coords[1][1], ch_coords[2][0], ch_coords[2][1],
                  float(data["Time"][1]), float(data["Time"][2]))

    while xi <= xi_max:
        while abs(s1 - s2) > acc and yi <= yi_max:
            yi += 0.1
            s1 = speed_eq(xi, yi, ch_coords[0][0], ch_coords[0][1], ch_coords[1][0], ch_coords[1][1],
                          float(data["Time"][0]), float(data["Time"][1]))
            s2 = speed_eq(xi, yi, ch_coords[1][0], ch_coords[1][1], ch_coords[2][0], ch_coords[2][1],
                          float(data["Time"][1]), float(data["Time"][2]))

        if abs(s1 - s2) <= acc and lthreshold <= s1 <= uthreshold and check_dist(xi, yi, ch_coords):
            res.append([round(xi, 4), round(yi, 4), round(s1, 2)])

        xi += 0.1

        yi = yi_min

        s1 = speed_eq(xi, yi, ch_coords[0][0], ch_coords[0][1], ch_coords[1][0], ch_coords[1][1],
                      float(data["Time"][0]), float(data["Time"][1]))
        s2 = speed_eq(xi, yi, ch_coords[1][0], ch_coords[1][1], ch_coords[2][0], ch_coords[2][1],
                      float(data["Time"][1]), float(data["Time"][2]))

    return res


# локация источников для множества волн
def localize(data, coords, l_threshold, h_threshold, width, events_n, acc):
    global WIDTH
    located_sources = []

    WIDTH = width

    i = 0

    while i < len(data):
        if data["Event"][i] == "le":
            index_le = i

            while i + 1 < len(data) and data["Event"][i + 1] == "ht":
                i += 1

            if i - index_le >= events_n:
                located_sources.extend(localize_wave(data[:][index_le:i + 1].reset_index(drop="true"), coords,
                                                     l_threshold, h_threshold, events_n, acc))
        i += 1

    return located_sources
