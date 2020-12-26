import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import time
import re
import warnings
import math
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import dendrogram

from IPython.display import clear_output, Markdown, display
from ipywidgets import IntProgress, HTML, VBox


from sklearn import preprocessing
from sklearn.cluster import KMeans,  AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KDTree



#Информация по датасету
def df_info(df):
    printmd('* Количество объектов : %d ' % (len(df)))
    printmd('* Количество признаков: %d' % (len(df.columns)))
    printmd('**Признаки:**')
    types = pd.DataFrame(df.dtypes.apply(lambda x: ' '+str(x)+' '))
    types.columns = [' Tипы  ']
    types['% of Nulls'] = (df.isna().sum()/df.count()).apply(lambda x: str(round(x,2)) + '%')
    display(types) 

    
#Первичный отбор признаков
def first_features_choice(df):
    printmd('Необходимо ли отобрать признаки из всех? (y/n)')
    val = ''
    while not val.lower() in ['y','n','yes','no']:
        val = input()
    if val.lower()[0] == 'y':
        f = features_choice_widget(df)
    else:
        f = df
    return f, val.lower()[0]


    
    
    
#Виджет для выбора столбца с индексом
def set_index_widget(df):
    d = df.columns.insert(0, 'None')
    dd = widgets.Dropdown(options=d,value=d[1], description='Column:',disabled=False)
    printmd('#### Установите название столбца с индексами (id)')
    display(dd)
    return dd


#Установка столбца с индексом 
def set_index_col(df, value):
    if value =='None':
        return df
    else:
        return df.set_index(value)

    
#Принт маркдаунов
def printmd(string):
    display(Markdown(string))
    

# Виджет указания столбцов с датой    
def date_preproc_widget(df):
    print('В признаках присутствуют даты?(y/n)')
    val = ''
    while not val.lower() in ['y','n','yes','no']:
        val = input()
    if val.lower()[0] == 'y':
        d = df.dtypes[(df.dtypes=='object')].index
        print('Укажите столбцы с датой:')
        cb = [0 for i in range(len(d))]
        for i in range(len(d)):
            cb[i] = widgets.Checkbox(options=df.dtypes[(df.dtypes=='object')].index, description=d[i],disabled=False)
            display(cb[i])  
    else:
        cb = 'no'
    return cb


#обработка дат
def date_preproc(dfs, d, dayfirst = False):
    if d!='no':
        df = dfs.copy()
        d = list(map(lambda x: x.description if x.value else 0 , d))
        for i in d:
            if i!=0:
                print('Столбец ',i)
                print('\nЧто делать с датой? \n1.Преобразовать\n2.Удалить')
                val = ''
                while not val in ['1','2']:
                    val = input()
                if val == '1':
                    print('\n1.Количество дней до текущей даты\n2.Количество дней до другой даты')    
                    val = ''
                    while not val in ['1','2']:
                        val = input()
                    if val=='1':
                        dd =  pd.datetime.now().date() 
                    else:
                        print('\nВведите дату (dd.mm.yy)')
                        p = False
                        while not p :
                            val = input()
                            try:
                                dd = pd.to_datetime(val, dayfirst = True).date()
                                p = True
                            except:
                                print('Неверный формат')
                    df[i] = abs((dd - df[i].apply(lambda x :pd.to_datetime(x, dayfirst = dayfirst).date())).apply(lambda x: x.days))
                else:
                    df = df.drop(i,1)
        return df
    else:
        return dfs

    
    
#распределения категориальных признаков
def cat_plots(df):
    printmd('**Распределение категориальных признаков**')
    cols = set(np.append(df.dtypes[df.dtypes=='object'].index.values,df.nunique()[df.nunique()<=8].index.values))
    if len(cols)==0:
        return 'Нет категориальных признаков'
    sns.set(style=('whitegrid'))
    sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
    if len(cols)==1:
        shape,cnt=1,1
    else:
        cnt = 5 if len(cols)%5 == 0 else(4 if len(cols)%4 == 0 else (3 if len(cols)%3==0 else (2 if len(cols)%2==0 else 3)))#количество графиков в ряд
        shape = math.ceil(len(cols)/cnt)
    fig, ax =plt.subplots(shape, cnt, figsize=(15, shape*4))
    k = 0
    j = 0
    for i in cols:
        if (shape==1)&(len(cols)==1):
            axes = ax
            fig.set_size_inches(10, 7)
        elif (shape==1)&(len(cols)>1):
            axes = ax[k]
        else:
            axes = ax[j,k]
        sns.countplot(df[i], ax = axes);
        axes.set_ylabel('')
        k+=1
        if k==cnt:
            k = 0
            j +=1

  

#удаление строк с большим числом null. Предлагает удалить, только если хотя бы у одной строки есть число NULL в более чем в половине признаков
def drop_nan_rows(dfs):
    df = dfs.copy()
    t = pd.DataFrame(df.isna().sum(axis=1).value_counts()).reset_index().sort_values('index', ascending=False).rename(columns = {'index':'Число NA',0:'Число объектов'})
    if t['Число NA'].max()>=(len(df.columns))//2:
        print('Количество пустых значений в объектах:')
        display(t.reset_index(drop=True))
        delete = ''
        printmd('-----')
        while not delete.lower() in ['y','n','yes','no']:
            delete = input('Удалить объекты с большим числом NA?(y/n) ')
        ind = False
        if (delete.lower()[0] == 'y'):
            printmd('**Выберите отсечение, по которому удалить объекты**')
            while ind==False:
                try:
                    tr = int(input('Удалить объекты, у которых число признаков с NA>='))
                    ind = True
                except:
                    print('\nОШИБКА, Введите целое число')
            d = df[df.isna().sum(axis=1)<tr]
            print('\nКоличество оставшихся объектов: %d'%len(d))
        else:
            d = df
    else:
        print('Нет объектов с большим числом NA')
        d = df
    return d                       
  
    
#удаление низковариативных признаков
def drop_unvariative_cols(dfs, c_var = 0.15):
    c = pd.DataFrame([], columns = ['f','var'])
    df = dfs.copy()
    le = preprocessing.LabelEncoder()#для проверки категориальных признаков на низкую вариативность
    printmd('Признаки с низким коэффициентом вариативности ' + '${std}\over{mean}$ :')
    for i in df.columns:
        if df[i].dtypes=='object':
            d = le.fit_transform(df[i].dropna())
        else:
            d = df[i]
        var =  d.std()/max(abs(d.mean()),1)
        if var<=c_var or (np.isnan(d.std())):    
            printmd('**'+str(i)+'** : %.3f' % var)
            c = c.append({'f':i, 'var':var}, ignore_index=True)
    if len(c) == 0:
        printmd('Нет признаков с низкой вариативностью')
        return df
    c_var = ' '
    while c_var[0] not in ['y','n']:
        printmd('Удалить признаки с низкой вариативностью? (y/n):')
        c_var = input()
    if c_var[0] == 'y':
        while c_var=='y':
            try:
                c_var = float(input('Удалить признаки, с коэфф. вариативности не больше: '))
            except:
                pass
        df = df.drop(c[c['var']<= c_var]['f'],1)
    return df 


#удаление и обработка аномалий
def extreme_values(dfs):
    df = dfs.copy()
    sns.set(style = ('whitegrid'))
    sns.set_context("notebook", font_scale = 1.3, rc = {"lines.linewidth": 2.5})
    s = df.shape[0]
    d = 0
    for i in df.columns:
        if (df[i].dtype != 'object') & (df[i].nunique() >= 6):
            iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
            tmp = df[(df[i]<df[i].quantile(0.25)-1.5*iqr) | (df[i] > df[i].quantile(0.75) + 1.5*iqr)].shape[0]
            if tmp>0:
                printmd('#### Выберите метод обработки аномалий для признаков')
                printmd('**%s**' % (i))
                f, ax = plt.subplots(figsize = (7, 3))
                sns.boxplot(df[i], ax = ax)
                ax.set_title('Распределение признака "%s".\n Аномальных значений: %d'%(i, tmp))
                display(f)
                cnt = 0
                print('Метод обработки аномальных значений:\n1.Удалить\n2.Ограничить\n3.Медиана(при предположении, что аномальные значения - ошибки в данных)\n4.Оставить как есть')
                while cnt == 0:
                    v = input('Выбор: ')
                    try:
                        val = int(v)
                    except:
                        val = 10
                    if val == 1:
                        d=1
                        df = df[(df[i] >= df[i].quantile(0.25) - 1.5 * iqr) & (df[i] <= df[i].quantile(0.75) + 1.5 * iqr)]
                        cnt = 1
                    elif val == 2:
                        df[i] = df[i].clip(lower = df[i].quantile(0.25) - 1.5 * iqr, upper = df[i].quantile(0.75) + 1.5 * iqr)
                        cnt=1
                    elif val == 3: 
                        df[i] = df[i].apply(lambda x: x if (x >= df[i].quantile(0.25) - 1.5 * iqr) & (x <= df[i].quantile(0.75) + 1.5 * iqr)  else df[i].median())
                        cnt = 1
                    elif val == 4:
                        cnt = 1
                        pass
                    else:
                        print('Неподходящее значение, введите другое: ')
            plt.close();
    if d==1:
        print('Удалено строк: %d '% (s - df.shape[0]))
    return df
        

    
    
#выводит процент пропусков в признаках, предлагает убрать, если в каких-то слишком мало значений,
#далее по оставшимся предлагает метод заполнения
def drop_nan_cols(dfs):
    df = dfs.copy()
    d = pd.DataFrame(round(df.isna().sum()/df.shape[0]*100,2))
    d.columns = ['%NA']
    d['Тип'] = df.dtypes
    if d['%NA'].max()>0: 
        print('Количество пропусков в признаках:\n')
        miss = d[d['%NA']!=0].sort_values('%NA',ascending = False)
        miss['%NA'] = miss['%NA'].apply(lambda x: str(x)+'%')
        display(miss)
        #print(d[d['%NA']!=0].sort_values('%NA',ascending = False).head(20))
        delete = ''
        while delete=='':
            try:
                delete = input('Удалить признаки с большим процентом NA? (y/n) ')
            except:
                pass
        if (delete.lower()[0] == 'y'):
            tr1 = float(input('\nУдалить признаки, с % NA>= '))
            df = df.drop(d[d['%NA']>=tr1].index,1)
            d = d.drop(d[d['%NA']>=tr1].index)
            print(d)
        #clear_output()
        
        for i in d[d['%NA']>0].index:
            printmd('#### Выберите метод заполнения пропусков для каждого признака')
            printmd('\nПризнак **%s**\n\nРаспределение:' % (i))
            if (d.loc[i]['Тип'] == 'object') or (df[i].nunique()<=10):
                q = df[i].value_counts(normalize=True)
                print(q)
                vals = {}
                printmd('---')
                print('Заполнить NA:')
                for j in range(df[i].nunique()):
                    if q.values[j] == max(q.values):
                        ss = ' (Мода)'
                    else:
                        ss = ''
                    print(str(j+1)+'.'+str(q.index[j])+ss)
                    vals.update({j+1:q.index[j]})
                print(str(df[i].nunique()+1)+'.Случайно (с весами из распределения)')
                print(str(df[i].nunique()+2)+'.Ручной ввод')
                cnt = 0
                while cnt == 0:
                    v = input()
                    try:
                        val = int(v)
                    except:
                        val = 10
                    if val in list(range(1, df[i].nunique() + 1)):####Поправить тут 
                        df[i] = df[i].fillna(vals[val])
                        cnt = 1
                    elif val == df[i].nunique()+1:
                        df[i] = df[i].apply(lambda x: np.random.choice(q.index,  p = q.values) if pd.isna(x) else x)
                        cnt = 1
                    elif val == df[i].nunique()+2:
        
                        g = input('Введите значение, которым заполнить пропуски: ')
                        try:
                            g = float(g)
                        except:
                            pass
                        df[i] = df[i].fillna(g)
                        cnt=1                      
                    else:
                        print('Неподходящее значение, введите другое: ')
                if df[i].dtypes == 'object':
                    df[i] = df[i].apply(lambda x: str(x).lower())
            elif (d.loc[i]['Тип'] != 'object') and (df[i].nunique()>=10):
                print(df[i].describe())
                printmd('---')
                print('Заполнить NA:\n \n1.Средним\n2.Медианой\n3.Ручной ввод\n')
                cnt = 0
                while cnt == 0:
                    v = input('Выбор: ')
                    try:
                        val = int(v)
                    except:
                        val = 10
                    if val == 1:
                        df[i] = df[i].fillna(df[i].mean())
                        cnt=1
                    elif val ==2:
                        df[i] = df[i].fillna(df[i].median())
                        cnt=1
                    elif val==3:
                        g = float(input('Введите значение, которым заполнить пропуски: '))
                        df[i] = df[i].fillna(g)
                        cnt=1
                    else:
                        print('Неподходящее значение, введите другое: ')
    else:
        print('Нет пропусков в признаках')
    
    return df    


#матрица корреляций + выводит признаки с высокой корреляцией    
def corr_diag(df):
    sns.set(style="white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(max(df.shape[1],10), max(df.shape[1]-2,8)))
    sns.heatmap(corr, mask=mask,  annot=True,  cmap="YlGnBu",  fmt=".2f",  vmax=.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .6});
    printmd('### Матрица корреляций признаков')
    plt.show()

    corrs = df.corr().reset_index().melt('index',var_name='index1', value_name='cor')
    corrs['val'] = corrs.apply(lambda x:' '.join(sorted([x['index'],x['index1']])),1)
    corrs = corrs.drop_duplicates('val')
    c = corrs[(corrs['index']!=corrs['index1'])&(abs(corrs.cor)>=0.75)]
    if c.shape[0]!=0:
        printmd('### Признаки с высокой корреляцией:')
    for i in c.values:
        printmd('**'+ i[0]+'** и **'+i[1]+'** =  %.5f' %i[2])

        
#Виджет выбора фич для кластеризации       
def features_choice_widget(df):
    cb = [0 for i in range(df.shape[1])]
    printmd('### Признаки, участвующие в кластеризации:')
    for i in range(df.shape[1]):
        cb[i] = widgets.Checkbox(options=df.dtypes[(df.dtypes=='object')].index, description=df.columns[i],disabled=False);
        cb[i].value = True
        display(cb[i]);
    return cb

#выбор
def features_choice(df, cb, val = 'y'):
    if val =='y':
        c = pd.Series(map(lambda x: x.description if x.value else 0 , cb))
        c = c[c!=0]
        return df[c]
    else:
        return df



#кодирование
def encoding(dfs):
    df = dfs.copy()
    cols1 = df.dtypes[df.dtypes=='object'].index.values
    cols2 = df.dtypes[(df.dtypes!='object')&(df.nunique()<=7)].index.values
    for i in cols2:
        printmd('**%s**, уникальных значений: %d'%(i,df[i].nunique()))
        p = input('Закодировать с помощью OneHot?(y/n) ')
        if (p.lower()=='y') or (p.lower()=='yes'):
            d = pd.get_dummies(df[i])
            k =df[i].unique()
            k.sort()
            d.columns = [i+'_%d'%j for j in k]
            df = df.join(d)
            df = df.drop(i,1)
        
    for i in cols1:
        printmd('**%s**, уникальных значений: %d'%(i,df[i].nunique()))
        print('Способ кодирования:\n1.LabelEncoder\n2.OneHot\n3.Удалить этот признак')
        cnt=0
        while cnt==0:
            v = input('Выбор: ')
            try:
                val = int(v)
            except:
                val = 10
            if val == 1:
                enc = preprocessing.LabelEncoder()
                df[i] = enc.fit_transform(df[i])
                cnt=1
            elif val ==2:
                df = df.join(pd.get_dummies(df[i]))
                df = df.drop(i,1)
                cnt=1
            elif val == 3:
                df = df.drop(i,1)   
                cnt=1
            else:
                print('Неподходящее значение, введите другое: ')
    return df


#масшатбирование
def scaling(dfs):
    cols = dfs.columns
    df = dfs.copy()
    print('Способ масштабирования данных: \n1.Стандартизация\n2.Нормализация[0,1]\n3.Не масштабировать')
    cnt=0
    while cnt==0:
            v = input('Выбор: ')
            try:
                val = int(v)
            except:
                val = 10
            if val == 1:
                sc = preprocessing.StandardScaler()
                df = sc.fit_transform(df)
                cnt=1
            elif val ==2:
                sc = preprocessing.MinMaxScaler(feature_range=(0, 1))
                df = sc.fit_transform(df)
                cnt=1
            elif val == 3:  
                cnt=1
            else:
                print('Неподходящее значение, введите другое: ')
    df = pd.DataFrame(df)
    df.columns = cols
    return df


##выбор метода кластеризации
def choice_method_widget():
    printmd('#### Выберите метод кластеризации')
    sel = widgets.Select(
    options=['K-means', 'Agglomerative' ,'DBSCAN'],
    value='K-means',
    rows=3,
    description='Метод:',
    disabled=False)
    display(sel)
    return sel


#подбор параметров кластеризации
def clustering_params(df, s):
    if s in ['K-means', 'Agglomerative']:
        clustering_graphs(df,s)
        printmd('### Посмотреть силуэты кластеров (y/n)?')
        if df.shape[0]>90000:
            printmd('Note: рассчёт силуэтов для больших датасетов может оказаться долгим (>100к примерно 30min.  >200к примерно 1h )?')
        val = 0
        while val not in ['y','yes','n','no']:
            val = input()
        if val[0]=='y':
            minmaxcl = [str(i) for i in range(2,50)]
            v = []
            while ((not all(i in minmaxcl for i in v)) | (len(v) != 2)):# проверка что введенные значения в диапазоне допустимых и их 2
                printmd('Введите диапазон числа кластеров, для которых рассмотреть силуэты $k_{min}$,$k_{max}$:')
                v = re.findall(r"[\d']+", input())
            v = list(map(int,v))
            silhoette(df,[min(v), max(v)], s)
        printmd('#### Выберите итоговое число кластеров:')
        k = 0
        while k<=1:
            try:
                k = int(input())
            except:
                pass
        return {"k":k} 
    elif s == 'DBSCAN':
        printmd('#### Построить график расстояний до k-ого соседа для оценки параметра $eps$?(y/n)')
        val = 0
        while val not in ['y','yes','n','no']:
            val = input()
        if val[0]=='y':
            clustering_graphs(df,s)
        else:
            #clear_output()
            pass
        printmd('#### Посмотреть силуэты кластеров (y/n)?')
        val = 0
        while val not in ['y','yes','n','no']:
            val = input()
        eps_list = ['']
        minPt_list =['']
        if val[0]=='y':
            while sum(map(lambda x: x.replace('.','',1).isdigit(),eps_list))!=len(eps_list):
                printmd('Введите значения $eps$ (через запятую)')
                eps_list = re.findall(r"\d+\.\d+", input())
            while sum(map(lambda x: x.isdigit(),minPt_list)) != len(minPt_list):
                printmd('Введите значения $minPt$ (через запятую)')
                minPt_list = re.findall(r"[\d']+", input())
            minPt_list = list(map(int,minPt_list))
            eps_list = list(map(float,eps_list))
            silhoette_db(df,{'eps':eps_list,'minPt':minPt_list})
        eps, minPt = 0,0
        while not(eps>0):
            printmd('Введите значение $eps$:')
            time.sleep(0.1)
            try:
                eps = float(input())
            except:
                pass
        while not(minPt>0):
            printmd('Введите значение $minPt$:')
            try:
                minPt = float(input())
            except:
                pass
        return {'eps':eps, 'minPt':minPt}
            

#графики для подбора параметров    
def clustering_graphs(df, s):
    minmaxcl = [str(i) for i in range(2,50)]
    val = ['0']
    if s == 'K-means':
        while ((not all(i in minmaxcl for i in val)) | (len(val) != 2)):# проверка что введенные значения в диапазоне допустимых и их 2
            printmd('Введите диапазон количества кластеров $k_{min}$,$k_{max}$:')
            val = re.findall(r"[\d']+", input())
            print('Неверно')
        clear_output()
        val = list(map(int,val))
        k_means_elbow(df, [min(val), max(val)])
    elif s == 'Agglomerative':
        agg_params(df, 3)
    elif s == 'DBSCAN':
        dbscan_params(df)

        
#построение гарфика до к-ого ближайшего соседа для DBSCAN       
def dbscan_params(df):
    k = df.shape[1]*2-1
    tree = KDTree(df)           
    dist, ind = tree.query(df, k=k+1) 
    f, ax = plt.subplots(figsize=(12,7))
    dist = pd.DataFrame(dist)[k].sort_values(ascending = False)
    sns.lineplot(x=range(len(dist)), y = dist, ax = ax)
    ax.set_ylabel('%d-distance'%k)
    ax.set_xlabel('')
    plt.show()
    printmd('Параметр $eps$ нужно определить как дистанцию до k-ого ($k = dim*2-1$) соседа, при которой возникает первая "впадина" на графике')
    printmd('Параметр minPt предлагается устанавливать как $2 \cdot dim$   ')
    printmd('<p><a href="http://www.ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf" target="_blank">Ссылка на статью</a></p>')

    
#метод локтя для k-means
def k_means_elbow(df, k_range):#запрашивать параметры в общей функции (k_range и метод(локтя или силуэты))
    progress = IntProgress(min=0, max=k_range[1], value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)
    inertias = {}
    progress.value = 0
    label.value = '%d/%d кластер'%(0,k_range[1])
    for i in range(k_range[0],k_range[1]+1):
        kmeans = KMeans(n_clusters=i, random_state=1234).fit(df)
        cluster_labels = kmeans.predict(df)
        inertias.update({i:kmeans.inertia_})        
        progress.value = i
        label.value =  '%d/%d кластер'%(i,k_range[1])
    progress.bar_style = 'success'
    #clear_output()
    sns.set(style=('whitegrid'))
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 2.5})
    printmd('### График изменения суммы квадратов расстояний в кластерах')
    f, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(x=list(inertias.keys()), y = list(inertias.values()), marker = 'o')
    ax.set_ylabel('inertia')
    ax.set_xlabel('k')
    plt.show()
  

#отрисовка дендограммы   
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)
 

#строит агломеративную кластеризацию и выводит дендограмму
def agg_params(df, levels):
    sns.set(style=('whitegrid'))
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(df)

    fig, ax = plt.subplots()
    ax.set_title('Дендограмма')
    fig.set_size_inches(15, 7)
    plot_dendrogram(model, truncate_mode='level', p=levels, ax = ax)
    plt.show()               
 

#силуэты для dbscan
def silhoette_db(df, param_grid):
    params = np.array(np.meshgrid(param_grid['eps'], param_grid['minPt'])).T.reshape(-1,2)
    m = 0
    progress = IntProgress(min=0, max=len(params)-1, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)
    silhoettes = {}
    for i in params:
        cluster_labels = DBSCAN(eps=i[0], min_samples=i[1]).fit_predict(df)
        if len(np.unique(cluster_labels))<=2:
            printmd("При $eps = %.2f$ и $minPt = %.2f$ выделился один кластер"%(i[0],i[1]))
        else:
            df_copy = df.copy()
            df_copy['labels'] = cluster_labels
            df_copy = df_copy[df_copy.labels !=-1]
            cluster_labels = cluster_labels[cluster_labels!=-1]
            silhouette_avg = silhouette_score(df_copy, cluster_labels)
            silhoettes.update({m:silhouette_avg})
            sample_silhouette_values = silhouette_samples(df_copy, cluster_labels)
            y_lower = 10
            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 5)
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(df_copy) + (m + 1) * 10])
            for j in range(len(np.unique(cluster_labels))):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == j]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(j) /(m+1))
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title('Средний коэфф. силуэта :%.3f'%silhouette_avg)
            ax1.set_xlabel("Коэффициент силуэта")
            ax1.set_ylabel("Номер кластера")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.suptitle(("Графики силуэта для eps = %.2f, minPt = %.2f" % (i[0],i[1])),fontsize=14, fontweight='bold')
        #clear_output()
        plt.show()
        #clear_output()
        progress.value = m
        m+=1
        label.value =  '%d/%d params'%(m,len(params))
    if len(silhoettes)>=1:
        printmd('#### Средние значения силуэтов для найденных кластеров:')
    for i in silhoettes.keys():
            printmd("*$eps = %.2f$, $minPt = %.2f$*  средний коэффициент силуэта: %5f "% (params[i][0],params[i][1],silhoettes[i]))
 

#силуэты
def silhoette(df, k_range, method):
    progress = IntProgress(min=0, max=k_range[1], value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)
    silhoettes = {}
    for i in range(k_range[0],k_range[1]+1):
        if method =='K-means':
            cluster_labels = KMeans(n_clusters=i, random_state=1234).fit_predict(df)
        elif method == 'Agglomerative':
            cluster_labels = AgglomerativeClustering(n_clusters=i).fit_predict(df)         
        silhouette_avg = silhouette_score(df, cluster_labels)
        silhoettes.update({i:silhouette_avg})
        sample_silhouette_values = silhouette_samples(df, cluster_labels)
        y_lower = 10
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10, 5)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(df) + (i + 1) * 10])
        for j in range(i):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == j]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(j) / i)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            progress.value = i
            label.value =  '%d/%d кластер'%(i,k_range[1])
        ax1.set_title('Средний коэфф. силуэта :%.3f'%silhouette_avg)
        ax1.set_xlabel("Коэффициент силуэта")
        ax1.set_ylabel("Номер кластера")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.suptitle(("Графики силуэта для k = %d" % i),fontsize=14, fontweight='bold')
    #clear_output()
    plt.show()
    #clear_output()
    for i in range(k_range[0],k_range[1]+1):
            printmd("*k = %d* средний коэффициент силуэта: %5f "% (i,silhoettes[i]))
            
            
#итоговая кластеризация с выбранными параметрами и методом
def clustering(df, s, params):
    if s == 'K-means':
        cluster_labels = KMeans(n_clusters=params['k'], random_state=1234).fit_predict(df)
    elif s =='Agglomerative':
        cluster_labels = AgglomerativeClustering(n_clusters=params['k']).fit_predict(df)  
    elif s == 'DBSCAN':
        cluster_labels = DBSCAN(eps=params['eps'], min_samples=params['minPt']).fit_predict(df)
    return cluster_labels


#Средние занчения в кластерах
def stat(df):
    printmd('### Средние значения характеристик в кластерах')
    mean = df.groupby('cluster').agg(['mean'])
    mean.columns = df.columns[:-1]
    return mean


#Выбор фич для графика
def features_choice_widget_g(df):
    df = df.drop('cluster',1)
    cb = [0 for i in range(df.shape[1])]
    printmd('#### Признаки, которые отобразить на графике:')
    for i in range(df.shape[1]):
        cb[i] = widgets.Checkbox(options=df.dtypes[(df.dtypes=='object')].index, description=df.columns[i],disabled=False);
        cb[i].value = True
        display(cb[i]);
    return cb


def features_choice_g(df, cb):
    c = pd.Series(map(lambda x: x.description if x.value else 0 , cb))
    c = c[c!=0]
    return c


#Выбор типа графика
def choice_graph_widget():
    sel = widgets.Select(options=['Обычный', 'RadarPlot'],value='Обычный',description='График:',disabled=False,rows = 2)
    display(sel)
    return sel


#масштабирование для визуализации
def scal_viz(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    d = data.drop(['cluster'],1)
    scaler = scaler.fit(d)
    q = pd.DataFrame(scaler.transform(d))
    q.columns = d.columns
    q['cluster'] = data.reset_index()['cluster']
    return q.groupby('cluster').apply(lambda x: (x.mean()-q.mean())/q.mean()*100).drop('cluster',1)


#визуализация отклонений от средних значений
def cluster_viz(dfc,  columns, colors, graph_type = '1', bold_line=None):
    columns  = list(columns)
    columns.append('cluster')
    df = dfc[columns]
    df =scal_viz(df).reset_index()
    if graph_type == 'RadarPlot':
        sns.set(style=('white'))
        sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.})
        categories=list(df)[1:]
        N = len(categories)
        ax = plt.subplot(111, polar=True)
        for i in range(df.cluster.nunique()):
            values=df.loc[i].drop('cluster').values.flatten().tolist()
            values += values[:1]
            angles = [n / float(N) * 2 * math.pi for n in range(N)]
            angles += angles[:1]        
            plt.gcf().set_size_inches(20, 8)
            ax.set_rlabel_position(-1)
            q = ax.plot(angles, values, linewidth=1.5, linestyle='solid', color = colors[i]);
        plt.xticks(angles[:-1], categories, color='black', size=13, rotation=90, ha='center')    
        plt.yticks([np.floor(min(df.min())),0,np.ceil(max(df.max()))], [str(np.floor(min(df.min())))+ '%','0',str(np.ceil(max(df.max()))) +'%'], color="grey", size=15)
        plt.ylim(np.floor(min(df.min()))-20,np.ceil(max(df.max())))
        ax.legend(['Кластер %d' % (i) for i in range(dfc.cluster.nunique())],loc='upper center', bbox_to_anchor=(0.5, 1.28),
              ncol=3, fancybox=True, shadow=True)
        ax.tick_params(axis='x', labelrotation=0,pad = 50)
        if bold_line!= None:
            values=df.loc[bold_line].drop('cluster').values.flatten().tolist()
            values += values[:1]
            angles = [n / float(N) * 2 * math.pi for n in range(N)]
            angles += angles[:1]      
            ax.plot(angles, values, linewidth=2, linestyle='solid', color = colors[bold_line]);
            plt.setp(ax.lines[-1],linewidth=3.5);

    elif graph_type == 'Обычный':
        sns.set(style=('whitegrid'))
        sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 1.5})
        fig, ax = plt.subplots(figsize=(20,8))
        for i in range(df.cluster.nunique()):
            q = sns.lineplot(x = 'index',y = i,  data = df.drop('cluster',1).T.reset_index(), ax = ax, color=colors[i])         
        ax.legend(['Кластер %d' % (i) for i in range(dfc.cluster.nunique())],loc='upper center', bbox_to_anchor=(0.5, 1.25),
              ncol=6, fancybox=True, shadow=True);
        ax.set(ylabel = '%',xlabel = '');
        if bold_line!= None:
            sns.lineplot(x = 'index',y = bold_line,  data = df.drop('cluster',1).T.reset_index(), ax = ax, color=colors[bold_line])
            plt.setp(ax.lines[-1],linewidth=3.5);
            
            
#выбор номера кластера           
def select_cluster(df,c=0):
    d = sorted(df.cluster.unique())
    dd = widgets.Dropdown(options=d,value=d[0], description='Кластер:',disabled=False)
    if c==1:
        printmd('##### Выберите кластер для выделения на графике: ')
    elif c==2:
        printmd('##### Выберите кластер для сравнения с остальными: ')
    elif c==0:
        printmd('##### Выберите кластер: ')
    display(dd)
    return dd


#пирог с размером кластеров
def pie_sizes(data):
    colors = dict()
    sns.set(style=('white'))
    sns.set_context("notebook", font_scale=.95, rc={"lines.linewidth": 2.5})
    f1, ax1 = plt.subplots(figsize=(10, 12))
    t = data.groupby('cluster').apply(lambda x: len(x)/data.shape[0]*100)# Размер кластера в процентах от размера выборки 
    printmd('## Размеры кластеров')
    patches, texts, autotexts = ax1.pie(t.values, labels=t.index, startangle=90,autopct='%1.2f%%',pctdistance=1.15,labeldistance=1.3);
    ax1.legend(['Кластер %d - %d ' % (i,len(data[data.cluster == i])) for i in range(data.cluster.nunique())],loc = 'lower right');
    for i in range(data.cluster.nunique()):
        colors.update({i:patches[i].get_facecolor()})
    for autotext in texts:
        autotext.set_color('white')
    for autotext in autotexts:
        autotext.set_fontsize(13)
    return colors
 
    
#сравнение распределений
def dist_comparing(data, main,  column):
    sns.set(style=('whitegrid'))
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)
    sns.distplot(data[data['cluster']==main][column], ax = ax, kde=True);
    sns.distplot(data[data['cluster']!=main][column], ax = ax, kde=True);
    ax.legend(['Кластер %d'%main,'Остальные кластеры']);

    
#выбор признака для сравнения    
def dist_col_widget(df):
    printmd('##### Выберите признак, по которому сравнить распределения: ')
    sel = widgets.Select(options=df.columns.drop('cluster',1),value=df.drop('cluster',1).columns[0],description='Признак:',disabled=False,rows = min(len(df.columns)-1,8))
    display(sel)
    return sel



def dist_comparing_all(data):
    printmd('#### Сравнение распределений: ')
    @interact
    def dist_comparing(Кластер = data.cluster.unique(),  Признак = data.drop('cluster',1).columns):
        sns.set(style=('whitegrid'))
        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 6)
        sns.distplot(data[data['cluster']==Кластер][Признак], ax = ax, kde=True);
        sns.distplot(data[data['cluster']!=Кластер][Признак], ax = ax, kde=True);
        ax.legend(['Кластер %d'%Кластер,'Остальные кластеры']);