from aelib import filtering


# получает список волн из pandas.DataFrame data
def get_waves(data):
    filtered_data = filtering.filter_signal_events_number(data, 4)
    waves = []
    i = 0

    while i < len(filtered_data):
        if filtered_data["Id"][i] == "le":
            index_le = i

            while i + 1 < len(filtered_data) and filtered_data["Id"][i + 1] == "ht":
                i += 1

            waves.append(filtered_data[:][index_le:i + 1].reset_index(drop="true"))

        i += 1

    return waves


# считает коэффициент Пирсона между парой волн
def count_pair_pearson(wave1, wave2):
    n = min(len(wave1), len(wave2))

    return round(wave1["A"][0:n].astype(float).corr(wave2["A"][0:n].astype(float)), 3)


# строит матрицу коэффициентов Пирсона для каждой пары волн в списке waves
def create_pearson_matrix(waves):
    matrix = []

    for wave1 in waves:
        wave_corr = []
        for wave2 in waves:
            wave_corr.append(count_pair_pearson(wave1, wave2))

        matrix.append(wave_corr)

    return matrix

# рассчитывает супер сигнал по списку волн
def create_super_signal(waves):
    super_signal = []

    min_n = len(waves[0])

    for k in range(0, len(waves)):
        if min_n < len(waves[k]):
            min_n = len(waves[k])

    for j in range(0, min_n):
        super_signal.append([0, 0])

        for i in range(0, len(waves)):
            super_signal[j][0] += float(waves[i]["A"][j])
            super_signal[j][1] += float(waves[i]["MSEC"][j])

        super_signal[j][0] = round(super_signal[j][0] / len(waves), 1)
        super_signal[j][1] = round(super_signal[j][1] / len(waves), 4)

    return super_signal


# рассчитывает супер сигналы по DataFrame
def create_super_signals(data):
    waves = get_waves(data)

    matrix = create_pearson_matrix(waves)
    super_signals = []

    for i in range(0, len(waves)):
        cluster = []
        for j in range(0, len(waves)):
            if matrix[i][j] >= 0.97:
                cluster.append(waves[j])

        if len(cluster) > 3:
            super_signals.append(create_super_signal(cluster))

    return super_signals
