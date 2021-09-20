
import pandas as pd
import numpy as np
import ppscore as pps
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import scipy.stats as ss

class Checker:
    """
    Осуществляет проверку на:
        1) кол-во пропущенных значений в каждом из признаков
        2) наличие константных и квазиконстантных признаков
        3) кол-ов классов в рамках целевой переменной
        4) корреляцию между признаками и целевой переменной

    Атрибуты: 
        data - датафрейм
    """
    def __init__(self, data):
                 #missing_edge=0,
                 #corr_method='pearson',
                 #corr_edge=0):
                 #kramer_edge=0):
        self.data = data
        self.test_frame = pd.DataFrame()
    
    def check_missing(self, missing_edge=0):
        """
        Проверка кол-ва пропущенных значений
        Args:
            missing_edge - порог {0..1} для отображения пропущенных значений
        Returns:
            Датафрейм с index = названия признаков, column = процент пропусков 
        """
        percent_missing = pd.DataFrame(self.data.isnull().mean().sort_values()*100, 
                                       columns=['missing'])
        return percent_missing[percent_missing['missing'] >= missing_edge]
    
    def constant_values(self):
        """
        Проверка константных и квазиконстантных признаков
        Returns:
            Cписок с именами признаков, являющихся конст-ми
        """
        label_encoding = preprocessing.LabelEncoder()
        objects = self.data.select_dtypes(include='object')
        numerical = self.data.select_dtypes(exclude='object')
        objects = objects.apply(label_encoding.fit_transform)
        self.test_frame = pd.concat([objects, numerical], axis=1)
        constant_filter = VarianceThreshold(threshold=0.01)
        constant_filter.fit_transform(self.test_frame)
        constant_columns = [col for col in self.test_frame.columns if col not in self.test_frame.columns[constant_filter.get_support()]]
        return constant_columns
    
    def disbalance_target(self):
        """
        Отображение соотнашения классов в рамках целевой переменной
        Returns:
            Гистограмма соотношения классов целевой переменной
        """
        self.test_frame = pd.DataFrame(self.data[(self.data.columns[len(self.data.columns)-1])])
        name = self.test_frame.columns[0]
        plt.figure(figsize=[5, 7])
        ax = sns.countplot(data=self.test_frame, 
                           x=name,
                           palette='mako',
                           edgecolor=sns.color_palette("dark", 3))
        plt.xticks(size=12)
        plt.xlabel(name, size=14)
        plt.yticks(size=12)
        plt.ylabel('count', size=12)
        plt.title('Count of categories', size=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        total = len(self.test_frame)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%\n'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='center')
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    def correlation(self, corr_edge=0, corr_method='pearson'):
        """
        Корреляция между признаками (Пирсон, Спирмен, Кендалл)
        Args:
            corr_edge - порог {0..1} для отображения коэф-ов корреляции
            corr_method - метод {‘pearson’, ‘kendall’, ‘spearman’} расчета коэф-ов корреляции    
        Returns:
            Датафрейм с index = названия признаков, column = значение коэф-та корр-ии
        """
        self.test_frame = self.data.corr(method=corr_method)
        self.test_frame = self.test_frame.unstack().sort_values(ascending=False).drop_duplicates()[1:][1:]
        return self.test_frame[(self.test_frame >= corr_edge) | (self.test_frame <= -corr_edge)]
    
    def ppscore(self, pps_edge=0):
        """    
        Корреляция между признаками (ppscore)
        Args:
            pps_edge - порог {0..1} для отображения коэф-ов корреляции    
        Returns:
            Датафрейм с index = названия признаков, column = значение коэф-та корр-ии
        """
        pps_values = pps.matrix(self.data)
        self.test_frame = pps_values[~(pps_values['x'] == pps_values['y'])][['x', 'y', 'ppscore']]
        self.test_frame = self.test_frame.sort_values(by='ppscore', ascending=False)
        self.test_frame = self.test_frame[self.test_frame['ppscore'] > pps_edge]
        return self.test_frame

    def kramer_corr(self, kramer_edge=0):
        """
        Корреляция между признаками (Крамер)
        Args:
            kramer_edge - порог {0..1} для отображения коэф-ов корреляции
        Returns:
            Датафрейм с index = названия признаков, column = значение коэф-та корр-ии
        """
        cat_columns =  self.data.select_dtypes(include='object').columns.tolist()
        shape = len(cat_columns)
        list_of_df =[]
        coefs = []

        for col_i, col_j in list(product(cat_columns, cat_columns)):
            df = pd.crosstab(self.data[col_i], self.data[col_j])
            list_of_df.append(df)

        def cramers_corrected_stat(confusion_matrix):
            """ 
            Расчет коэффициентов Крамера
            calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
            """
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            coef_ = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

            return coef_
    
        for i in list_of_df:
            coef = cramers_corrected_stat(i)
            coefs.append(coef)
        self.test_frame = pd.DataFrame(data=np.array(coefs).reshape(shape,shape), index=cat_columns, columns=cat_columns)
        self.test_frame = self.test_frame.round(3)
        self.test_frame = self.test_frame.unstack().sort_values(ascending=False)
        self.test_frame = self.test_frame[self.test_frame >= kramer_edge]
        self.test_frame = self.test_frame.drop_duplicates()[1:]
        return self.test_frame

