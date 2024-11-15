import numpy as np
import pandas as pd
import locate_all_signals as loc


DataExp = pd.read_csv(r'Данные для эксперимента.txt', delimiter=' ').to_numpy()
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
loc.location(DataExp, Dd, 3140, visual=True)
