from sklearn.cluster import DBSCAN
import math as m
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from random import randint
import pandas as pd
def dist(x1,x2,y1,y2):
    return (m.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))
dataset = pd.read_csv(r'Rectаngel_Hsu-Nelson-4D_all.txt.txt', delimiter='\t')
#sensors = pd.read_csv(r'sensors_Triangel_Hsu-Nelson_coords.txt', delimiter='\t')

#dataset.columns = cols
dataset.columns = dataset.columns.str.replace(' ','')

#dataset = dataset[(dataset['Номерканалa'] != 13) & (dataset['Номерканалa'] != 14)]
#dataset = dataset[dataset['Амплитуда,дБ'] > 30].reset_index(drop = True)
dataset.insert(0, "Тип", np.zeros(len(dataset)))
dataset.insert(1, "Время,мкс", np.zeros(len(dataset)))
i = 0

dataset = dataset.replace(r'\s','', regex=True)
dataset = dataset.drop(columns = ['Типинформации(АЭ)','Аппаратныефлаги', 'Времясостарта,с','Амплитуда,АЦП', 'Амплитуда,мкВ', 'Ампл./Выбросы,мкВ', 'Энергия,MARSE', 'Дата,дд.мм.гг'],axis = 1)

max_dist = 3100
s = 3230
DT1X_MAX = 0.0015*max_dist/s #s
FHCDT = 0.03 #s



while i < len(dataset):
    t_le = datetime.strptime((dataset['Время,чч:мм:сс'][i] + '.' + str(dataset["+мкс"][i]).replace('.', '')[:6]), "%H:%M:%S.%f")
    delta_le = timedelta(hours=t_le.hour, minutes=t_le.minute, seconds=t_le.second,microseconds=t_le.microsecond)
    dataset.iloc[i, 1] = delta_le
    #t_ht = datetime.strptime(dataset['Время,чч:мм:сс'][i] + '.' + str(dataset['+мкс'][i]).replace('.', '')[:6],'%H:%M:%S.%f')
    #delta_ht = timedelta(hours=t_ht.hour, minutes=t_ht.minute, seconds=t_ht.second,microseconds=t_ht.microsecond)

    #if delta_ht.total_seconds() - delta_le.total_seconds() > 0.0002:
        #le_i = i
        #dataset.iloc[i, 0] = 1

    i += 1
dataset.sort_values(by=['Время,мкс'], inplace=True)

dataset.reset_index(inplace= True, drop = True)
file = pd.DataFrame(columns=['LE', 'Ht1', 'Ht2', 'Ht3', 'T1', 'T2', 'T3', 'AVG_AMP', 'LE_DSET'] )

i = 0
le_i = 0
dset = []
while i < len(dataset):
    if abs(dataset.iloc[le_i, 1].total_seconds() - dataset.iloc[i, 1].total_seconds()) > DT1X_MAX:
        if i-1 >= le_i:
            if abs(dataset.iloc[i-1, 1].total_seconds() - dataset.iloc[i, 1].total_seconds()) > FHCDT:
                if i - le_i <= 4:
                    dataset.iloc[le_i:i+1, 0] = 'ev'
                dataset.iloc[i, 0] = 'le'
                dset.append(i)
                le_i = i

            else:
                if i - le_i <= 4:
                    dataset.iloc[le_i:i+1, 0] = 'ev'
                dataset.iloc[i, 0] = 'ev'
        else:
            if i - le_i <= 4:
                dataset[le_i:i+1, 0] = 'ev'

            le_i = i
            dataset.iloc[i, 0] = 'le'
            dset.append(i)
    elif i-1 >= le_i:
        if abs(dataset.iloc[i-1, 1].total_seconds() - dataset.iloc[i, 1].total_seconds()) > FHCDT:
            dataset.iloc[i, 0] = 'ev'
        else:
            if i - le_i > 3:
                dataset.iloc[i, 0] = 'ev'
            else:
                dataset.iloc[i, 0] = 'ht'
    else:
        if i - le_i > 3:
            dataset.iloc[i, 0] = 'ev'
        else:
            dataset.iloc[i, 0] = 'ht'
    i += 1
dataset.insert(0, "dset", np.arange(len(dataset)))
dataset.iloc[0, 1] = 'le'
dataset.to_csv('Rectangel_tech_v_ventil_all.csv', index = False, sep=' ')
dataset = dataset[dataset['Тип'] != 'ev'].reset_index(drop = True)
#dataset.to_csv('Triаngel_tech_v_ventil_all.txt', index=False, sep=' ')
print(dataset)
i = 0
while i < len(dataset)-1:
    if dataset.iloc[i, 1] == 'le':
        file.loc[len(file.index)] = [int(dataset['Номерканалa'][i]), int(dataset['Номерканалa'][i+1]), int(dataset['Номерканалa'][i+2]),
                                     int(dataset['Номерканалa'][i+3]), np.round(abs(dataset.iloc[i, 2].total_seconds() - dataset.iloc[i+1, 2].total_seconds())*1000,6),
                                     np.round(abs(dataset.iloc[i, 2].total_seconds() - dataset.iloc[i+2, 2].total_seconds())*1000,6),
                                     np.round(abs(dataset.iloc[i, 2].total_seconds() - dataset.iloc[i+3, 2].total_seconds())*1000,6),
                                     np.round(np.mean(dataset.iloc[i:i+4,5]), 6), str(dset[i%4]) + ';']
    i = i + 4
file[['LE', 'Ht1', 'Ht2', 'Ht3']] = file[['LE', 'Ht1', 'Ht2', 'Ht3']].astype('int64')
file[['T1', 'T2', 'T3', 'AVG_AMP']] = file[['T1', 'T2', 'T3', 'AVG_AMP']].astype('float64')
print(file)
file.to_csv('Rectаngel_Hsu-Nelson-4D_all_filtered.csv', index=False, sep=' ')