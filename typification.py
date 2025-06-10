import pandas as pd
import numpy as np
from collections import OrderedDict
from aelib import filtering
from sklearn.cluster import AgglomerativeClustering
from aelib import location
from pandas.api.types import is_float_dtype
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as mt
from sklearn.metrics import davies_bouldin_score

def b_score(N,A,c):
    return(20*(c - mt.log(N,10))/A)

def classification(df, n):
    kmeans = AgglomerativeClustering(n_clusters=3, linkage='average', metric='cosine').fit(df)

    #print(len(set(dataset['Damage Mechanism'].unique()) & set(np.unique(kmeans.labels_)))/len(kmeans.labels_))
    #for i in range(5):
        #data = dataset[dataset['Damage Mechanism'] == i]
        #fig, ax = plt.subplots()
        #ax.scatter(data['A'], data['F'])
        #plt.show()
    labels = kmeans.labels_
    max_mean_amp = -1
    min_mean_amp = 1000
    max_ind = -1
    min_ind = -1
    for j in range (3):
        A = df['A'][labels == j][n:].mean()
        if (A > max_mean_amp):
            max_mean_amp = A
            max_ind = j
        if (A < min_mean_amp):
            min_mean_amp = A
            min_ind = j
    if (len(df[labels == min_ind])>1):
        kmeans1 = AgglomerativeClustering(n_clusters=2, linkage='average', metric='cosine').fit(df[labels == min_ind])
        labels1 = kmeans1.labels_
    else:
        labels1 = [0]
    if (len(df[labels == max_ind]) > 1):
        kmeans2 = AgglomerativeClustering(n_clusters=2, linkage='average', metric='cosine').fit(df[labels == max_ind])
        labels2 = kmeans2.labels_ + np.full(np.size(kmeans2), 2)
    else:
        labels2 = [0]
    n = 0
    j = 0
    for k in range (np.size(labels)):
        if (labels[k] == min_ind):
            labels[k] = labels1[n]
            n += 1
        elif (labels[k] == max_ind):
            labels[k] = labels2[j]
            j += 1
        else:
            labels[k] = 4
    return labels
colorsArray = {'Отслоение': "#2C40A2", "Трение": '#9932CC', "Разрыв": '#DC143C', "Растяжение": '#0E9417', "Трещина": '#7FFF00'}
colorsArray2 = {'Debonding': "#2C40A2", "Friction": '#9932CC', "Break": '#DC143C', "Pullout": '#0E9417', "Crack": '#7FFF00'}
translate = {'Interface Debonding': 'Отслоение', "Fiber Friction": "Трение", "Fiber Break": "Разрыв",
             "Fiber Pullout": "Растяжение", "Matrix Crack": "Трещина"}
translate2 = {0: 'Отслоение', 1: "Трение", 2: "Разрыв", 3: "Растяжение", 4: "Трещина"}
days = ['06.11', '07.11', '08.11', '09.11', '31.10', '01.11', '03.11', '04.11', '05.11', '12.11', '13.11', '14.11',
        '15.11', '16.11', '17.11']

dataset = pd.read_csv(r'Training_dataset_20230130.csv', delimiter=';')
coun = dataset['Damage Mechanism'].value_counts()

coef = []
b_values = []

for l in dataset['Damage Mechanism'].unique():
    coef.append(coun[0] / coun[l])
s = coun[0]
for l in range(len(coef)):
    coef[l] = coef[l]/s*100
    #print(coef[l]*coun[dataset['Damage Mechanism'].unique()[l]])
res = {dataset['Damage Mechanism'].unique()[i]: coef[i] for i in range(len(dataset['Damage Mechanism'].unique()))}
print(res)
temp = [93.8, 94]
EN = [103025.7, 862762]
print((1+(temp[1]-temp[0])/temp[0])**4 - 1)
print(EN[1]/EN[0])

for j in range(1,2):
    df = pd.read_csv(str(j) + '.csv', decimal=',')
    df['HHMMSS'] = pd.to_datetime(df['HHMMSS'], format='%H:%M:%S')
    w1 = 0
    w2 = 100
    while w2 < len(df):
        df_w = df[w1:w2]
        w1+=1
        w2+=1
        b_values.append(b_score(100, max(df_w['A']), 5))
    #df = df.loc[(df['HHMMSS'].dt.time <= pd.Timestamp('08:00:00').time()) | (df['HHMMSS'].dt.time >= pd.Timestamp('20:00:00').time())]
    #df = df.loc[df['HHMMSS'].dt.time >= pd.Timestamp('20:00:00').time()]
    defects = ['friction','cracking', 'debonding', 'break']
    #print(dataset['Damage Mechanism'].value_counts())
    #df[['CNTS','E(TE)','A','D']] = df[['CNTS','E(TE)','A','D']].astype('float64')
    df['CNTS'] = df['CNTS']/df['D']
    df = df.rename(columns = {'CNTS':'F'})
    df = df.rename(columns = {'E(TE)':'E'})
    df['F'] = df['F'].astype('float64')
    a = (df['F']<0.6)
    df = df[a]

    l = len(df)
    object_df = df


    df = pd.concat([df[['A','E','F']],dataset[['A','E','F']]])


    labels = classification(df,len(object_df))
    cur_defects = ['Отслоение','Трение','Разрыв','Разрыв','Отслоение']
    unique_labels = set(labels)

    lab = labels[l:]
    mp = 0
    for i in unique_labels:
        d = dataset[lab == i]
        for k in d['Damage Mechanism'].unique():
            r = d['Damage Mechanism'].value_counts().loc[k]*res[k]/len(d)
            if r > mp:
                cur_defects[i] = k
                mp = r
    cur_defects = pd.DataFrame(cur_defects, columns = ['def'])
    #cur_defects['def'] = cur_defects['def'].replace(translate)
    cur_colors = cur_defects['def'].replace(translate).replace(colorsArray)
    object_df.reset_index(inplace=True)
    cur_defects = pd.unique(cur_defects['def'])

    old_labels = labels[len(object_df):]
    #labels = labels[:len(object_df)]
    colors = np.array([cur_colors[l] for l in labels])
    #labels = pd.DataFrame(labels, columns = ['def']).replace(cur_defects['def'])
    #labels['def'] = str(labels['def'])
    names=[]
    TC = []
    for i in set(colors):
        if i in colors[:len(object_df)]:
            d = dataset[colors[len(object_df):] == i]
            x = []
            for k in d['Damage Mechanism'].unique():
                r = d['Damage Mechanism'].value_counts().loc[k] * res[k] / len(d)
                x.append(np.floor(r * 100))
            x.sort(reverse=True)
            TC.append(x)
    colors = colors[:len(object_df)]
    q = 0
    for i in set(colors):

        names.append(str(list(colorsArray2.keys())[list(colorsArray2.values()).index(i)]) + '(' + str(sum(colors == i)) + ')')
        q+=1

    unique_labels = set(labels)
    #print(unique_labels)
    fig, ax = plt.subplots(layout='tight')
    def update(i):
        #clear the axis each
        ax.clear()
        # replot things
        t = ax.scatter(object_df[:i+1]["F"], object_df[:i+1]["A"], c=colors[:i+1])

        # reformat things
        plt.xlabel('Frequency, MHz')
        plt.ylabel('Amplitude, dB')
        plt.title(days[j - 1])
        plt.xticks(fontsize=12)
        recs = []
        for n in set(colors[:i+1]):
            recs.append(plt.Rectangle((0, 0), 1, 1, facecolor=n))
        ax.legend(handles=recs, labels=names[:i+1], loc='center left', bbox_to_anchor=(1, 0.5),
                  prop={'size': 12})
        plt.xlim([0, 0.6])
        plt.ylim([40, 90])

    #ani = animation.FuncAnimation(fig,update,frames=len(object_df)-1,interval=10)
    #ani.save(str(days[j-1])+'scatter.gif', writer='pillow', fps=10)
    t = ax.scatter(object_df["F"], object_df["A"], c=colors, s=5)
    plt.xlabel('Frequency, MHz')
    plt.ylabel('Amplitude, dB')
    plt.title(days[j - 1])
    recs = []
    stats = pd.DataFrame(colors, columns=['Тип'])
    stats['Тип'] = stats['Тип'].replace({v: k for k, v in colorsArray.items()})
    stats['День'] = '1'
    energy = []
    for d in np.unique(colors):
        energy.append(100*object_df['E'][colors == d].sum()/object_df['E'].sum())

    stats_E = pd.DataFrame(data=np.reshape(energy, (1, len(energy))), columns=pd.DataFrame(np.unique(colors)).replace(({v: k for k, v in colorsArray.items()})),
                            index=['1'])
    #stats_E['Defect type'] = cur_defects
    #stats_E['Day'] = '1'

    for i in set(colors):
        recs.append(plt.Rectangle((0, 0), 1,1, facecolor=i))
    ax.legend(handles=recs, labels=names, loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'size': 12})
    plt.xticks(fontsize=12)
    plt.xlim([0, 0.6])
    plt.ylim([40, 90])
    plt.savefig(days[j-1] + '.jpg', bbox_inches='tight')
    plt.show()

    sns.displot(data=stats, x='День', hue='Тип', stat='percent', multiple="stack", aspect= .5, color=np.unique(colors))
    plt.title("Доля числа событий дефекта (%)")
    plt.show()

    stats_E.plot(kind='bar', stacked=True, color=np.unique(colors))
    plt.title("Доля энергии дефекта (%)")
    plt.show()

    sns.lineplot(data=b_values)
    plt.show()
    plt.close()
