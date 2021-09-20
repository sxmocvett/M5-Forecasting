
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from itertools import product
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

class DetectOutliers:

    def __init__(self, data, multiple_columns=None):
        self.test_frame = data.copy()
        self.multiple_columns = multiple_columns

    def feature(self, name_col):
        feature = self.test_frame[name_col].values.reshape(-1, 1)
        return feature

    def output(self, name_col, mask, drop, name_method):
        mask = pd.Series(mask, name=f"anomalies_{name_col}_{name_method}")
        if drop:
            mask.replace({False: 1, True: np.nan}, inplace=True)
        else:
            mask.replace({False: 1, True: -1}, inplace=True)
        mask.name = f"anomalies_{name_col}_{name_method}"
        self.test_frame.insert(self.test_frame.columns.get_loc(name_col)+1, mask.name, mask.values)
        return self.test_frame

    def outlier_detect_iqr(self, name_col, threshold=1.5, drop=False, name_method='iqr'):
        feature = self.test_frame[name_col]
        IQR = feature.quantile(0.75) - feature.quantile(0.25)
        Lower_fence = feature.quantile(0.25) - (IQR * threshold)
        Upper_fence = feature.quantile(0.75) + (IQR * threshold)
        mask = (feature < Lower_fence) | (feature > Upper_fence)
        mask = mask.ravel()
        self.test_frame = self.output(name_col, mask, drop, name_method)
        return self.test_frame

    def outlier_detect_mean_std(self, name_col, threshold=3, drop=False, name_method='mean_std'):
        Lower_fence = self.feature(name_col).mean() - threshold*self.feature(name_col).std()
        Upper_fence = self.feature(name_col).mean() + threshold*self.feature(name_col).std()
        mask = (self.feature(name_col) < Lower_fence) | (self.feature(name_col) > Upper_fence)
        mask = mask.ravel()
        self.test_frame = self.output(name_col, mask, drop, name_method)
        return self.test_frame

    def lof(self, name_col, n_neighbors=20, drop=False, name_method='lof'):
        model = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof = model.fit_predict(self.feature(name_col))
        self.test_frame = self.output(name_col, lof, drop, name_method)
        return self.test_frame

    def svm(self, name_col, nu=0.1, drop=False, name_method='svm'):
        model = OneClassSVM(nu=nu)
        svm = model.fit_predict(self.feature(name_col))
        self.test_frame = self.output(name_col, svm, drop, name_method)
        return self.test_frame

    def iso_forest(self,
                   name_col,
                   n_estimators=100,
                   contamination=0.01,
                   max_features=0.01,
                   bootstrap=True,
                   drop=False,
                   name_method='iso_forest'):
        model = IsolationForest(n_estimators=n_estimators,
                                contamination=contamination,
                                max_features=max_features,
                                bootstrap=bootstrap)
        iso_forest = model.fit_predict(self.feature(name_col))
        self.test_frame = self.output(name_col, iso_forest, drop, name_method)
        return self.test_frame

    def dbscan(self, name_col, eps=0.01, drop=False, name_method='dbscan'):
        model = DBSCAN(eps=eps, min_samples=2)
        dbscan = model.fit_predict(self.feature(name_col))
        self.test_frame = self.output(name_col, dbscan, drop, name_method)
        return self.test_frame

    def all_models(self, name_col):
        check = input('Ввести гиперпараметры вручную?')
        if check == 'Да' or check == 'да':
            print('IQR')
            threshold_iqr = float(input('Введите treshold: '))
            print('mean_std')
            threshold_mean_std = float(input('Введите treshold: '))
            print('lof')
            n_neighbors = float(input('Введите количество соседей: '))
            print('svm')
            nu = float(input('Введите nu: '))
            print('isolation forest')
            n_estimators = float(input('Введите n_estimators: '))
            contamination = float(input('Введите contamination: '))
            max_features = float(input('Введите max_features: '))
            print('dbscan')
            eps = float(input('Введите eps: '))
        else:
            threshold_iqr = 1.5
            threshold_mean_std = 3
            n_neighbors = 20
            nu = 0.1
            n_estimators = 100
            contamination = 0.01
            max_features = 0.01
            eps = 0.01
        iqr = self.outlier_detect_iqr(name_col, threshold_iqr).filter(like='iqr')
        mean_std = self.outlier_detect_mean_std(name_col, threshold_mean_std).filter(like='mean_std')
        lof = self.lof(name_col, n_neighbors).filter(like='lof')
        svm = self.svm(name_col, nu).filter(like='svm')
        iso_forest = self.iso_forest(name_col, n_estimators, contamination, max_features).filter(like='iso_forest')
        dbscan = self.dbscan(name_col, eps).filter(like='dbscan')
        methods = pd.concat([iqr, mean_std, lof, svm, iso_forest, dbscan], axis=1)
        return methods

    def models_one_col(self, col):
        self.all_models(col)
        return self.test_frame

    def models_all_col(self):
        for col in self.multiple_columns:
            self.all_models(col)
        return self.test_frame

