from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import numpy as np

def variance_filter(X, threshold=0.5):
    """剔除低方差特征
    方差:衡量一个特征取值的离散程度。
    如果某个特征在所有样本上的值几乎不变(方差接近 0),那么这个特征几乎没有区分能力，属于“低信息量”特征。
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    return X_filtered, selector

def standardize(X_train, X_test):
    """标准化，返回 scaler 和变换后的数据"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def select_kbest(X_train, y_train, X_test, k=2000):
    """互信息特征选择"""
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector