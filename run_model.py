import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts

from libs.dataset import DataSet
from libs.training import Training
from libs.graphics import Graphics



def main(dataframe):
    x_features, y_labels = ds.get_characteristic(_dataframe=dataframe)
    gphs.graphics(x_features, y_labels, obo=False)                            #Grafico las trece caracteristicas

    # Separo entre datos de entrenamiento y datos de test
    x_features_training, x_features_test, y_labels_training, y_labels_test = tts(x_features, y_labels, test_size=0.2)
    learning_rate, num_epochs, display_step, rows, columns = trng.get_hiperparameters(x_features_training)
    #Creo mis placeholders para las features y caracteristica y mi variables.
    X_features_ph, Y_labels_ph = trng.get_placeholders(rows=rows, colms=columns)
    theta, theta_cero = trng.get_variables(colms=columns)

    # Obtengo la media de mi funcion de costo
    y_labels_training_tmp = tf.cast(y_labels_training, tf.float32)   #Recuerda que se te presento el error de floar64 proveniente del MatMul
    mean_J = trng.get_mean_j(y_labels_training_tmp, X_features_ph)

    # Optimizer
    optimizer = trng.tftGDO()
    train_cost = []
    ep = []

    # Entrenamiento del modelo.
    ep, train_cost, th, th0 = trng.model(ep, train_cost, X_features_ph, x_features_training, Y_labels_ph, y_labels_training, theta, theta_cero);
    gphs.graphics_with_plot(ep,train_cost)

    # Aplicamos el test
    t_accuracy = trng.apply_test(x_features_test, y_labels_test)
    print('>> Accuracy: {:.4f} %\n'.format(t_accuracy[0]*100))
    print('\tTesting Completed.')


if __name__=='__main__':
    ruta_xlsx = 'dataset/boston.xlsx'
    ruta = 'dataset/boston.csv'
    ds = DataSet(ruta)
    trng = Training(lr=0.52)  # Para la normalizacion con mean: lr=0.009, y con max_value lr=0.52 o simplemente se deja vacia la casilla
    gphs = Graphics()                                                        # Inicializo mi clase dataset
    ds.convert_dataset_from_xlsx_to_csv(ruta_xlsx)
    dataset = ds.get_dataset()
    #Creamos el dataframe
    dataframe = ds.data_to_norm(pd.DataFrame(dataset), mean=False)   # Primero se convierte a un dataframe de pandas>> dataframe = pd.DataFrame(dataset)
                                                                    # Para normalizacioncon mean: mean=True, con max_value false o vacio
    main(dataframe)
