import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts


class DataSet():
    def __init__(self, _ruta):
        self.ruta = _ruta

    def convert_dataset_from_xlsx_to_csv(self, _ruta_xlsx):
        _dataset_xlsx = pd.read_excel(_ruta_xlsx)
        _dataset_xlsx.to_csv(self.ruta, encoding='utf-8')
        print('[OK]    From xlsx to csv.')

    def get_dataset(self):
        _dataset = pd.read_csv(self.ruta, encoding='utf-8')
        self.dataset = _dataset.replace(np.nan, '0')
        return self.dataset

    def data_to_norm(self, _dataframe):
        for _dims in _dataframe:
            if _dims == 'Cost':
                break
            else:
                _value_max = _dataframe[_dims].max()
                _value_min = _dataframe[_dims].min()
                #print('{}>> Valor max: {}, Valor min: {}'.format(_dims, _value_max, _value_min) )
                _dataframe[_dims] = (_dataframe[_dims] - _value_min)/(_value_max - _value_min)
        self.dataframe = _dataframe
        print('[OK]    DataFrame normalizado')
        return self.dataframe

    def get_characteristic(self, _dataframe):
        _trainig_data = _dataframe.shape[0] * _dataframe.shape[1]
        self.x_features = _dataframe[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT' ]].dropna(axis=0, how='any')
        self.y_labels = _dataframe[['Cost']].dropna(axis=0, how='any')
        print('[OK]    Get characteristic')
        return self.x_features,  self.y_labels
