from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import make_column_selector
import numpy as np
import pandas as pd
import re


# преобразование признака mileage
def _kmkg2kmpl(value):
    try:
        if isinstance(value, str):
            parts = value.split()
            parts[0] = float(parts[0])
            if parts[1] == 'km/kg':
                parts[0] *= 1.3
            return parts[0]
        return value
    except:
        return np.nan


# преобразование крутящего момента
def _moment(value: str):
    try:
        if 'nm' in value:
            return float(value[:value.index('nm')])
        if 'kgm' in value:
            return float(value.replace('kgm', '')) * 9.80665
        if '(' in value:
            return float(value.split('(')[0])
        return float(value)
    except:
        return np.nan


# преобразование оборотов
def _rpm(value: str):
    try:
        value = value.replace(',', '').replace('~', '-').strip()
        value = re.sub(r"\+/-.+", '', value)
        value = re.sub(r"\D*$", '', value)
        if '-' in value:
            parts = list(map(float, value.split('-')))
            return (parts[0] + parts[1]) / 2
        return float(value) if len(value) != 0 else np.nan
    except:
        return np.nan


# разбиение колонки torque на две с преобразованием
def _torque(value: str):
    try:
        if pd.isnull(value):
            return [np.nan, np.nan]
        
        value = value.strip().lower()
        if value == '':
            return [np.nan, np.nan]
        
        parts = []
        if '@' in value:
            parts = value.split('@', 2)
        elif 'at' in value:
            parts = value.split('at')
        elif '/' in value:
            parts = value.split('/')

        if len(parts) != 0:
            return [_moment(parts[0]), _rpm(parts[1])]
        
        return [_moment(value), np.nan]
    except:
        return [np.nan, np.nan]
    

def _remove_units(value):
    try:
        if isinstance(value, str):
            return value.replace(r'\s.*', '', regex=True).replace('', np.nan).astype('float')
    except:
        pass
    return np.nan


def get_ohe_columns(X):
    return make_column_selector(dtype_include=object)(X) + ['seats']


class BasePreprocessing(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self


class MileagePreprocessing(BasePreprocessing):
    def transform(self, X):
        return X[self.feature_names_in_].apply(lambda x: x.apply(_kmkg2kmpl))
    

class NamePreprocessing(BasePreprocessing):
    def transform(self, X):
        return X[self.feature_names_in_].apply(lambda x: x.str.split(' ').str[0])
    

class RemoveUnitsPreprocessing(BasePreprocessing):
    def transform(self, X):
        return X[self.feature_names_in_].apply(lambda x: x.apply(_remove_units))
    

class TorquePreprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def get_feature_names_out(self, input_features=None):
        return self._out_features
    
    def transform(self, X):
        tmp = pd.DataFrame(X.torque.apply(_torque).to_list(), columns=['torque', 'max_torque_rpm'])
        df = pd.concat([X.drop(['torque'], axis=1), tmp], axis=1)
        self._out_features = df.columns.tolist()
        return df
