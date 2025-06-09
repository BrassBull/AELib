from datetime import datetime, timedelta


# фильтр по событиям
def filter_ev(data):
    return data.loc[data["Id"].isin(["le", "ht"])]


# фильтр по пороговым значениям амплитуды
def filter_threshold(data, l_threshold, u_threshold):
    filtered_data = data.loc[data["A"].astype(float) >= l_threshold]
    filtered_data = filtered_data.loc[filtered_data["A"].astype(float) <= u_threshold]

    return filtered_data.reset_index(drop=True)


# фильтр по числу датчиков, зафиксировавших сигнал
def filter_signal_events_number(data, num):
    id_list = data["Id"]
    mask = [True] * len(data)
    index_le = -1
    events_number = 0
    i = 0

    for val in id_list:
        if val == "le":
            if 0 < events_number < num - 1:
                while events_number >= 0:
                    mask[index_le + events_number] = False
                    events_number -= 1
            index_le = i
            events_number = 0
        elif val == "ht":
            if index_le > -1:
                events_number += 1
            else:
                mask[i] = False
        else:
            mask[i] = False
            if 0 < events_number < num - 1:
                while events_number >= 0:
                    mask[index_le + events_number] = False
                    events_number -= 1
            events_number = 0
            index_le = -1

        i += 1

    return data[mask].reset_index(drop=True)


# фильтр от электромагнитных шумов
def filter_electromagnetic(data):
    filtered_data = filter_signal_events_number(data, 4)
    mask = [True] * len(filtered_data)

    i = 0

    while i < len(filtered_data):
        if filtered_data["Id"][i] == "le":
            index_le = i

            t_le = datetime.strptime(filtered_data["HHMMSS"][i] + "." + filtered_data["MSEC"][i].replace('.', '')[:6],
                                     "%H:%M:%S.%f")
            delta_le = timedelta(hours=t_le.hour, minutes=t_le.minute, seconds=t_le.second,
                                 microseconds=t_le.microsecond)

            while i + 1 < len(filtered_data) and filtered_data["Id"][i + 1] != "le":
                i += 1

            t_ht = datetime.strptime(filtered_data["HHMMSS"][i] + "." + filtered_data["MSEC"][i].replace('.', '')[:6],
                                     "%H:%M:%S.%f")
            delta_ht = timedelta(hours=t_ht.hour, minutes=t_ht.minute, seconds=t_ht.second,
                                 microseconds=t_ht.microsecond)

            if delta_ht.total_seconds() - delta_le.total_seconds() < 0.0002:
                for k in range(0, i - index_le + 1):
                    mask[index_le + k] = False

        i += 1

    return filtered_data[mask].reset_index(drop=True)

