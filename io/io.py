import pandas as pd


def open_file(filename: str):
    with open(filename, "r", encoding='utf8') as file:
        params = list(filter(None, file.readline().replace("\n", '').replace("\ufeff", '').split(' ')))

        line = ""

        while not ("le" in line or "ht" in line or "ev" in line):
            line = file.readline().lower()

        data = pd.DataFrame([], columns=params)

        line_data = list(filter(None, line.replace("\n", ' ').replace(",", ".").split(' ')))
        line_df = pd.DataFrame(line_data, index=params[0:len(line_data)]).T

        data = pd.concat([data, line_df], ignore_index=True)

        for l in file:
            line_data = list(filter(None, l.lower().replace("\n", ' ').replace(",", ".").split(' ')))
            line_df = pd.DataFrame(line_data, index=params[0:len(line_data)]).T
            data = pd.concat([data, line_df], ignore_index=True)

    return data
